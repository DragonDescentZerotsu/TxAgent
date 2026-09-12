"""Hash-bound, source-only diagnostic edits to indexed representative cards.

Apply before cumulative family pools and neighbor selection so replacements and
earlier/later placement are recomputed. Never changes the serialized base index.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .progressive_assay_reasoning import _card_id, _card_surface, stable_analog_id

_CORRECTABLE_FIELDS = {"assay_context", "endpoint_type", "reported_value", "reported_units",
                       "species_context", "qualifying_conditions", "support_text"}


def _validate_reassignment(row: dict[str, Any]) -> None:
    """Require an original-document receipt for a corrected record association."""
    target = row.get("target_canonical_smiles")
    provenance = row.get("identity_provenance") or {}
    if (not isinstance(target, str) or not target.strip()
            or target == row.get("canonical_smiles")
            or not all(isinstance(provenance.get(k), str) and provenance[k].strip()
                       for k in ("document_path", "document_sha256", "locator"))):
        raise ValueError("reassignment requires a distinct indexed target and original-document provenance")
    path = Path(provenance["document_path"])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != provenance["document_sha256"]:
        raise ValueError("identity reassignment original document missing or hash mismatch")
    if "field_updates" in row:
        _validate_correction(row)
    if "target_level" in row or "target_family" in row:
        if (not isinstance(row.get("target_level"), int) or row["target_level"] <= 1
                or not isinstance(row.get("target_family"), str) or not row["target_family"].strip()):
            raise ValueError("reassignment placement cannot grant voter membership")


def _validate_correction(row: dict[str, Any]) -> None:
    updates = row.get("field_updates") or {}
    if (not updates or not set(updates) <= _CORRECTABLE_FIELDS
            or not all(isinstance(v, str) and v.strip() for v in updates.values())):
        raise ValueError("corrections require nonempty source fields; identity and family are immutable")
    if "target_level" in row or "target_family" in row:
        if (not isinstance(row.get("target_level"), int) or row["target_level"] <= 1
                or not isinstance(row.get("target_family"), str) or not row["target_family"].strip()):
            raise ValueError("correction placement cannot grant voter membership")
    if "support_text" in updates:
        provenance = row.get("support_text_provenance") or {}
        if not all(isinstance(provenance.get(k), str) and provenance[k].strip()
                   for k in ("document_path", "document_sha256", "locator")):
            raise ValueError("support is immutable without a hash-bound original document and locator")
        path = Path(provenance["document_path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != provenance["document_sha256"]:
            raise ValueError("support correction original document missing or hash mismatch")


def surface_sha256(surface: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        surface, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def load_record_review(path: Path, *, task: str, index_sha256: str,
                       levels: list[dict[str, Any]]) -> dict[str, Any]:
    review = json.loads(path.read_text())
    if review.get("version") != "indexed_record_review.v1":
        raise ValueError("unsupported indexed record review version")
    if review.get("task") != task or review.get("base_index_sha256") != index_sha256:
        raise ValueError("record review task/base index mismatch")
    if review.get("selection_policy", "assay_diverse.v1") not in {
        "assay_diverse.v1", "endpoint_diverse_delta.v1",
    }:
        raise ValueError("unknown record review selection policy")
    families = {int(row["level"]): row["endpoint_group"] for row in levels}
    seen = set()
    for row in review["decisions"]:
        if row["card_id"] in seen:
            raise ValueError("duplicate record review card")
        seen.add(row["card_id"])
        if row["action"] not in {"move", "exclude", "keep", "correct", "reassign"} or not row.get("reason"):
            raise ValueError("invalid record review decision")
        if row["action"] == "correct":
            _validate_correction(row)
        if row["action"] == "reassign":
            _validate_reassignment(row)
        if (row["action"] == "move" or
                (row["action"] in {"reassign", "correct"} and "target_level" in row)) and (
            int(row["target_level"]) <= 1
            or families.get(int(row["target_level"])) != row["target_family"]
        ):
            raise ValueError("record review cannot grant voter membership or unknown family")
        if row["action"] != "keep" and row["source_family"] == families[1]:
            raise ValueError("diagnostic review cannot edit actual voter cards")
    return review


def _add_reviewed_targets(index: dict[str, Any], decisions: dict[str, Any]) -> dict[str, Any]:
    """Derive missing corrected molecules through the existing index/identity code.

    Only an explicitly authorized, document-bound reassignment may add a target;
    callers cannot supply fingerprints or leakage keys. No evidence is invented.
    """
    present = {m["canonical_smiles"] for m in index["molecules"]}
    missing = {d["target_canonical_smiles"] for d in decisions.values()
               if d["action"] == "reassign"} - present
    if not missing:
        return index
    from .task_workflows.evidence_library import _standardize_molecule_task, fingerprint_metadata
    from .starling.new_task_retrieval_identity import annotate_index, query_identity

    if index.get("fingerprint") != fingerprint_metadata():
        raise ValueError("new identity target requires the standard index fingerprint contract")
    if len(index.get("fingerprints", [])) != len(index["molecules"]):
        raise ValueError("new identity target requires aligned index fingerprints")
    molecules, fingerprints = list(index["molecules"]), list(index["fingerprints"])
    ids = {m["molecule_chembl_id"] for m in molecules}
    for smiles in sorted(missing):
        rows = [d for d in decisions.values() if d.get("target_canonical_smiles") == smiles]
        if any(d.get("allow_new_target") is not True or not d.get("target_inchi_key") for d in rows):
            raise ValueError("identity reassignment target must exist uniquely or explicitly permit a verified new target")
        mid = "REVIEW_" + hashlib.sha256(smiles.encode()).hexdigest()[:24].upper()
        _, canonical, inchi, fp, identity = _standardize_molecule_task((mid, smiles, True))
        if (not fp or canonical != smiles or mid in ids
                or any(d["target_inchi_key"] != inchi for d in rows)):
            raise ValueError("new identity target canonical structure/InChIKey mismatch")
        molecule = dict(molecule_chembl_id=mid, canonical_smiles=canonical,
                        standard_inchi_key=inchi, molecule_identity=identity,
                        fingerprint_source="canonical_smiles", groups=[], n_evidence_rows=0)
        source = index.get("source") or {}
        contract = source.get("retrieval_identity_contract")
        if contract:
            task = source["retrieval_identity_task"]
            annotate_index({"molecules": [molecule], "source": {}}, task,
                           {smiles: query_identity(task, smiles, contract)}, contract)
        molecules.append(molecule)
        fingerprints.append(fp)
        ids.add(mid)
    return {**index, "molecules": molecules, "fingerprints": fingerprints}


def _split_complete_aggregate(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild one-record rows only when every underlying record is represented.

    Never carry mixed measurements, names, scopes or free-text summaries to a
    corrected molecule. Materialize the old visible surface first so an
    untouched sibling keeps its card identity and source qualifiers.
    """
    from .evidence_contract import attach_minimal_evidence
    examples = row.get("source_record_examples") or []
    if (row.get("source_record_count") != len(examples) or not examples or
            any(not ex.get("support_text") or int(ex.get("evidence_family_level") or 0) == 1
                for ex in examples)):
        raise ValueError("aggregate split requires complete nonvoter source examples with support")
    result = []
    for original in examples:
        surface = _card_surface(original, row)
        example = {**original, **{
            {"endpoint": "endpoint_type", "reported_unit": "reported_units", "species": "species_context"}.get(k, k): v
            for k, v in surface.items()
        }}
        single = {k: row[k] for k in ("molecule_chembl_id", "canonical_smiles", "standard_inchi_key",
            "assay_chembl_id", "assay_tier", "endpoint_group", "group_id", "evidence_source") if k in row}
        single.update(source_record_examples=[example], source_record_count=1,
            source_support_texts=[surface["support_text"]], source_molecule_names=[],
            standard_type=surface["endpoint"], standard_relation="", standard_value=surface["reported_value"],
            standard_units=surface["reported_unit"], assay_description=surface["support_text"],
            organism=surface["species"], target_pref_name=surface["assay_context"],
            evidence_scope={key: [example[key]] for key in ("assay_context", "species_context", "qualifying_conditions") if example[key]},
            activity_comment="One source record from a completely represented reviewed aggregate",
            evidence_role="unspecified", transferability="not_assessed",
            review_split_complete_aggregate=True)
        assert _card_surface(example, single) == surface
        result.append(attach_minimal_evidence(single))
    return result


def apply_record_review(index: dict[str, Any], review: dict[str, Any]
                        ) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions = {row["card_id"]: row for row in review["decisions"]}
    for decision in decisions.values():
        if decision["action"] == "correct":
            _validate_correction(decision)
        if decision["action"] == "reassign":
            _validate_reassignment(decision)
    index = _add_reviewed_targets(index, decisions)
    targets = {d["target_canonical_smiles"] for d in decisions.values() if d["action"] == "reassign"}
    reassignment_sources = {d["canonical_smiles"] for d in decisions.values() if d["action"] == "reassign"}
    target_molecules = {}
    for molecule in index["molecules"]:
        if molecule.get("canonical_smiles") in targets:
            target_molecules.setdefault(molecule["canonical_smiles"], []).append(molecule)
    if any(len(target_molecules.get(target, [])) != 1 for target in targets):
        raise ValueError("identity reassignment target must exist uniquely in the indexed molecule table")
    smiles = {row["canonical_smiles"] for row in decisions.values()}
    molecules = {str(row["molecule_chembl_id"]): row for row in index["molecules"]
                 if row.get("canonical_smiles") in smiles}
    evidence = dict(index["evidence_by_molecule_group"])
    matches: Counter = Counter()
    transfers = []
    for molecule_id, molecule in molecules.items():
        groups = evidence.get(molecule_id, {})
        replacement = {}
        analog = stable_analog_id(molecule)
        for group_id, rows in groups.items():
            retained_rows = []
            expanded_rows = []
            for row in rows:
                if molecule["canonical_smiles"] not in reassignment_sources:
                    expanded_rows.append(row)
                    continue
                examples = row.get("source_record_examples") or []
                assigned = [decisions.get(_card_id(analog, _card_surface(ex, row))) for ex in examples]
                moving = [d for d in assigned if d and d["action"] == "reassign"]
                mixed = moving and (len(moving) != len(examples) or
                                    len({d["target_canonical_smiles"] for d in moving}) != 1)
                if mixed and all(d.get("allow_complete_aggregate_split") is True for d in moving):
                    for ex, decision in zip(examples, assigned):
                        if not decision or decision["action"] != "reassign":
                            continue
                        surface = _card_surface(ex, row)
                        for field in _CORRECTABLE_FIELDS:
                            key = {"endpoint_type": "endpoint", "reported_units": "reported_unit", "species_context": "species"}.get(field, field)
                            if (not str(ex.get(field) or "").strip() and surface[key]
                                    and field not in decision.get("field_updates", {})):
                                raise ValueError("aggregate split requires explicit corrections for inherited card fields")
                    expanded_rows.extend(_split_complete_aggregate(row))
                else:
                    expanded_rows.append(row)
            for row in expanded_rows:
                row_decisions = ([decisions.get(_card_id(analog, _card_surface(ex, row)))
                                  for ex in row.get("source_record_examples") or []]
                                 if molecule["canonical_smiles"] in reassignment_sources else [])
                reassigned = [d for d in row_decisions if d and d["action"] == "reassign"]
                if reassigned:
                    # Moving only part of an aggregate would retain another test
                    # article's row-level summaries; fail closed instead.
                    if (len(reassigned) != len(row_decisions)
                            or len({d["target_canonical_smiles"] for d in reassigned}) != 1):
                        raise ValueError("reassignment requires the complete aggregate row to share one target")
                retained_examples = []
                for example in row.get("source_record_examples") or []:
                    surface = _card_surface(example, row)
                    card_id = _card_id(analog, surface)
                    decision = decisions.get(card_id)
                    if decision is None:
                        retained_examples.append(example)
                        continue
                    if (decision["canonical_smiles"] != molecule["canonical_smiles"]
                            or decision["surface_sha256"] != surface_sha256(surface)
                            or decision["source_family"] != surface["evidence_family"]):
                        raise ValueError(f"review payload mismatch: {card_id}")
                    matches[card_id] += 1
                    if decision["action"] == "exclude":
                        continue
                    if (decision["action"] == "move" or
                            (decision["action"] in {"reassign", "correct"} and "target_level" in decision)):
                        example = {**example, "evidence_family": decision["target_family"],
                                   "evidence_family_level": decision["target_level"]}
                    if decision["action"] in {"correct", "reassign"} and decision.get("field_updates"):
                        updates = decision["field_updates"]
                        example = {**example, **updates}
                    retained_examples.append(example)
                if retained_examples:
                    updated_row = {**row, "source_record_examples": retained_examples}
                    if reassigned:
                        target = target_molecules[reassigned[0]["target_canonical_smiles"]][0]
                        updated_row = copy.deepcopy(updated_row)
                        if updated_row.get("review_split_complete_aggregate"):
                            updated_row = _split_complete_aggregate(updated_row)[0]
                        # These aggregate mirrors must not retain a corrected
                        # test-article name after the examples move together.
                        for old_example, new_example in zip(row['source_record_examples'], retained_examples):
                            old_support = old_example.get('support_text')
                            new_support = new_example.get('support_text')
                            if (old_support and new_support and old_support != new_support
                                    and not updated_row.get("review_split_complete_aggregate")):
                                if 'source_support_texts' in updated_row:
                                    updated_row['source_support_texts'] = [
                                        text.replace(old_support, new_support)
                                        for text in updated_row['source_support_texts']
                                    ]
                                text_fields = updated_row.get('minimal_evidence', {}).get('text', {})
                                for key, text in text_fields.items():
                                    if isinstance(text, str):
                                        text_fields[key] = text.replace(old_support, new_support)
                        for key in ("molecule_chembl_id", "canonical_smiles", "standard_inchi_key"):
                            if key in target:
                                updated_row[key] = target[key]
                            else:
                                updated_row.pop(key, None)
                        if "minimal_evidence" in updated_row:
                            updated_row["minimal_evidence"]["molecule"].update(
                                id=target["molecule_chembl_id"], canonical_smiles=target["canonical_smiles"])
                        if updated_row.get("review_split_complete_aggregate"):
                            names = [reassigned[0]["resolved_compound_name"]] if reassigned[0].get("resolved_compound_name") else []
                            updated_row["source_molecule_names"] = names
                            updated_row["minimal_evidence"]["molecule"]["names"] = names
                        transfers.append((str(target["molecule_chembl_id"]), group_id, updated_row))
                    else:
                        retained_rows.append(updated_row)
            if retained_rows:
                replacement[group_id] = retained_rows
        if replacement:
            evidence[molecule_id] = replacement
        else:
            evidence.pop(molecule_id, None)
    missing = sorted(set(decisions) - set(matches))
    if missing:
        raise ValueError(f"reviewed cards absent from base index: {missing}")
    for target_id, group_id, row in transfers:
        groups = dict(evidence.get(target_id, {}))
        groups[group_id] = [*groups.get(group_id, []), row]
        evidence[target_id] = groups
    touched = set(molecules) | {target_id for target_id, _, _ in transfers}
    updated_molecules = [
        {**m, "groups": sorted(evidence.get(str(m["molecule_chembl_id"]), {})),
         "n_evidence_rows": sum(len(rows) for rows in evidence.get(str(m["molecule_chembl_id"]), {}).values())}
        if transfers and str(m["molecule_chembl_id"]) in touched else m
        for m in index["molecules"]
    ]
    group_indices: dict[str, list[int]] = {}
    for position, molecule in enumerate(updated_molecules):
        for group in evidence.get(str(molecule["molecule_chembl_id"]), {}):
            group_indices.setdefault(group, []).append(position)
    return {**index, "molecules": updated_molecules, "evidence_by_molecule_group": evidence,
            "group_to_molecule_indices": group_indices,
            "record_review_selection_policy": review.get("selection_policy", "assay_diverse.v1")}, {
        "status": "ok", "scope": "indexed representative cards before cumulative selection",
        "selection_policy": review.get("selection_policy", "assay_diverse.v1"),
        "matches_by_card": dict(matches),
        "matched_occurrences_by_action": dict(Counter({
            action: sum(matches[key] for key, row in decisions.items() if row["action"] == action)
            for action in ("move", "exclude", "keep", "correct", "reassign")
        })),
    }
