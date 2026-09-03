"""Build the leakage-filtered Skin causal-panel seed index and MiniMol descriptor."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
from typing import Any

import pandas as pd

from tools.chembl_tool.common.json_utils import read_jsonl, sha256_file, write_jsonl_atomic
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_features import (
    candidate_order_sha256,
    load_retrieval_index,
)
from tools.chembl_tool.paper_experiments.skin_causal_panel_seed.skin_contract import (
    CONTRACT_VERSION,
    SOURCE_GROUP_ID,
    compile_skin_causal_panel_rows,
    stable_json_sha256,
)


DEFAULT_CANONICAL_ROOT = Path("data/artifacts/starling/skin_reaction/canonical_sources/canonical_sensitization_v3")
DEFAULT_CANONICAL_EVIDENCE = Path(
    "outputs/paper/molecular_evidence_agent/evidence/"
    "skin_reaction_starling_sensitization_canonical_v3/starling_skin_reaction_evidence.jsonl"
)
DEFAULT_BASE_DESCRIPTOR = Path(
    "outputs/paper/minimol_retrieval_features_skin_canonical_v3_record_supported_v2_valid_verified/"
    "scaffold/descriptors/skin_reaction__starling_full_mechanism.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/paper/skin_causal_panel_seed_v1_scaffold_valid_deepseek_v4_pro")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact = build_skin_seed_artifacts(
        canonical_root=args.canonical_root,
        canonical_evidence=args.canonical_evidence,
        base_descriptor=args.base_descriptor,
        output_root=args.output_root,
    )
    print(json.dumps(artifact, indent=2), flush=True)
    return 0


def build_skin_seed_artifacts(
    *,
    canonical_root: Path,
    canonical_evidence: Path,
    base_descriptor: Path,
    output_root: Path,
) -> dict[str, Any]:
    direct_path = canonical_root / "direct_records.parquet"
    aop_path = canonical_root / "aop_records.parquet"
    for path in (direct_path, aop_path, canonical_evidence, base_descriptor):
        if not path.is_file():
            raise FileNotFoundError(path)

    canonical_rows = read_jsonl(canonical_evidence)
    cards, compiler_stats = compile_skin_causal_panel_rows(
        pd.read_parquet(direct_path).to_dict(orient="records"),
        pd.read_parquet(aop_path).to_dict(orient="records"),
        canonical_rows,
    )
    if not cards:
        raise ValueError("Strict Skin causal-panel compiler produced no evidence cards")

    descriptor = json.loads(base_descriptor.read_text(encoding="utf-8"))
    base_index_path = _resolve_descriptor_path(
        base_descriptor, str(descriptor["base_index_path"])
    )
    with base_index_path.open("rb") as handle:
        original_index = pickle.load(handle)
    original_order_hash = candidate_order_sha256(original_index)
    if original_order_hash != str(descriptor.get("candidate_order_sha256") or ""):
        raise ValueError("Frozen MiniMol descriptor does not match its base index")

    seed_index, materialization = materialize_causal_group(original_index, cards)
    if candidate_order_sha256(seed_index) != original_order_hash:
        raise AssertionError("Causal-group materialization changed MiniMol candidate order")

    evidence_dir = output_root / "evidence" / "skin_reaction_starling_causal_panel_seed_v1"
    descriptor_dir = output_root / "retrieval_features" / "scaffold" / "descriptors"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    full_evidence_path = evidence_dir / "starling_skin_reaction_evidence.jsonl"
    cards_path = evidence_dir / "causal_panel_cards.jsonl"
    index_path = evidence_dir / "starling_skin_reaction_neighbor_index.pkl"
    meta_path = evidence_dir / "starling_skin_reaction_neighbor_index.meta.json"
    descriptor_path = descriptor_dir / "skin_reaction__starling_causal_panel_seed_v1.json"
    audit_path = output_root / "source_build_audit.json"

    write_jsonl_atomic(full_evidence_path, [*canonical_rows, *cards])
    write_jsonl_atomic(cards_path, cards)
    with index_path.open("wb") as handle:
        pickle.dump(seed_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    seed_descriptor = deepcopy(descriptor)
    seed_descriptor.update(
        {
            "base_index_path": str(index_path),
            "experiment": "skin_reaction__starling_causal_panel_seed_v1",
            "causal_panel_contract": CONTRACT_VERSION,
            "source_descriptor": str(base_descriptor),
        }
    )
    descriptor_path.write_text(
        json.dumps(seed_descriptor, indent=2) + "\n", encoding="utf-8"
    )
    # Force the common runtime validator to check order and embedding dimensions.
    loaded = load_retrieval_index(descriptor_path)
    if SOURCE_GROUP_ID not in loaded["group_to_molecule_indices"]:
        raise AssertionError("Materialized descriptor cannot see the causal-panel group")

    source_inputs = {
        str(path): sha256_file(path)
        for path in (direct_path, aop_path, canonical_evidence, base_descriptor, base_index_path)
    }
    audit = {
        "type": "skin_causal_panel_seed_source_build.v1",
        "contract_version": CONTRACT_VERSION,
        "source_inputs_sha256": source_inputs,
        "compiler": compiler_stats,
        "materialization": materialization,
        "invariants": {
            "candidate_order_unchanged": True,
            "candidate_order_sha256": original_order_hash,
            "existing_group_membership_unchanged": True,
            "existing_group_evidence_unchanged": True,
            "minimol_descriptor_runtime_validation": "passed",
        },
        "paths": {
            "full_evidence_jsonl": str(full_evidence_path),
            "causal_cards_jsonl": str(cards_path),
            "index_pkl": str(index_path),
            "descriptor_json": str(descriptor_path),
        },
    }
    meta = {
        "index_version": str(seed_index.get("version") or ""),
        "n_evidence_rows": len(canonical_rows) + len(cards),
        "n_index_molecules": len(seed_index["molecules"]),
        "groups": sorted(seed_index["group_to_molecule_indices"]),
        "source": seed_index["source"],
        "source_stats": {"compiler": compiler_stats, "materialization": materialization},
        "paths": audit["paths"],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def materialize_causal_group(
    base_index: dict[str, Any],
    cards: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add one group to a frozen index without changing its molecule order."""
    if SOURCE_GROUP_ID in base_index.get("group_to_molecule_indices", {}):
        raise ValueError(f"Base index already contains {SOURCE_GROUP_ID}")
    output = deepcopy(base_index)
    existing_groups = sorted(base_index["group_to_molecule_indices"])
    existing_membership_hash = stable_json_sha256(
        {group: base_index["group_to_molecule_indices"][group] for group in existing_groups}
    )
    existing_evidence_hash = stable_json_sha256(
        {
            molecule_id: {
                group: rows
                for group, rows in groups.items()
                if group in existing_groups
            }
            for molecule_id, groups in base_index["evidence_by_molecule_group"].items()
        }
    )
    by_parent: dict[str, list[int]] = {}
    for index, molecule in enumerate(output["molecules"]):
        identity = molecule.get("molecule_identity") or {}
        key = str(identity.get("parent_inchi_key") or "").strip()
        if not key:
            normalized = normalize_molecule_identity(str(molecule.get("canonical_smiles") or ""))
            key = normalized.parent_inchi_key or normalized.parent_smiles
        if key:
            by_parent.setdefault(key, []).append(index)

    added_indices: list[int] = []
    excluded_heldout_or_missing = 0
    for card in cards:
        key = str(card.get("parent_inchi_key") or "").strip()
        candidate_indices = by_parent.get(key, [])
        molecule_index = next(
            (
                index
                for index in candidate_indices
                if str(output["molecules"][index]["molecule_chembl_id"])
                == str(card["molecule_chembl_id"])
            ),
            None,
        )
        if molecule_index is None:
            excluded_heldout_or_missing += 1
            continue
        molecule = output["molecules"][molecule_index]
        molecule_id = str(molecule["molecule_chembl_id"])
        if molecule_id != str(card["molecule_chembl_id"]):
            raise ValueError(f"Causal card molecule id mismatch for parent {key}")
        group_rows = output["evidence_by_molecule_group"][molecule_id]
        if SOURCE_GROUP_ID in group_rows:
            raise ValueError(f"Duplicate causal group for molecule {molecule_id}")
        group_rows[SOURCE_GROUP_ID] = [deepcopy(card)]
        molecule["groups"] = sorted([*molecule["groups"], SOURCE_GROUP_ID])
        molecule["n_evidence_rows"] = int(molecule["n_evidence_rows"]) + 1
        added_indices.append(molecule_index)
    if not added_indices:
        raise ValueError("All causal cards were excluded from the held-out-filtered index")
    output["group_to_molecule_indices"][SOURCE_GROUP_ID] = sorted(added_indices)
    output["version"] = f"{base_index.get('version', 'starling')}.causal_panel_seed_v1"
    output["source"] = {
        "type": "starling_skin_causal_panel_seed.v1",
        "base_source": deepcopy(base_index.get("source") or {}),
        "causal_panel_contract": CONTRACT_VERSION,
        "zero_parent_overlap": bool((base_index.get("source") or {}).get("zero_parent_overlap")),
    }

    new_membership_hash = stable_json_sha256(
        {group: output["group_to_molecule_indices"][group] for group in existing_groups}
    )
    new_evidence_hash = stable_json_sha256(
        {
            molecule_id: {
                group: rows
                for group, rows in groups.items()
                if group in existing_groups
            }
            for molecule_id, groups in output["evidence_by_molecule_group"].items()
        }
    )
    if new_membership_hash != existing_membership_hash:
        raise AssertionError("Existing retrieval-group membership changed")
    if new_evidence_hash != existing_evidence_hash:
        raise AssertionError("Existing evidence rows changed")
    return output, {
        "n_input_cards": len(cards),
        "n_materialized_cards": len(added_indices),
        "n_excluded_heldout_or_missing_cards": excluded_heldout_or_missing,
        "n_existing_index_molecules": len(output["molecules"]),
        "existing_group_membership_sha256": existing_membership_hash,
        "existing_group_evidence_sha256": existing_evidence_hash,
    }


def _resolve_descriptor_path(descriptor_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = descriptor_path.parent / path
    return candidate if candidate.exists() else path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--canonical-evidence", type=Path, default=DEFAULT_CANONICAL_EVIDENCE)
    parser.add_argument("--base-descriptor", type=Path, default=DEFAULT_BASE_DESCRIPTOR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
