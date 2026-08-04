"""Optional, identity-safe context for reasoning over coverage-selected analogs.

Neighbor selection and LLM context are deliberately separate contracts.  This
module never changes which molecules were retrieved.  The default ``standard``
profile is an exact no-op; ``coverage_aware`` adds legacy anonymous Morgan-feature
and approximate query-atom diagnostics.  The visible-only ``coverage_mmp_ledger``
profile instead combines Morgan feature complementarity with the existing
``mmp_structure_compare`` MCS/MMP text for each selected analog.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


STANDARD_NEIGHBOR_CONTEXT = "standard"
COVERAGE_AWARE_NEIGHBOR_CONTEXT = "coverage_aware"
COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT = "coverage_mmp_ledger"
NEIGHBOR_CONTEXT_PROFILES = (
    STANDARD_NEIGHBOR_CONTEXT,
    COVERAGE_AWARE_NEIGHBOR_CONTEXT,
    COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT,
)
COVERAGE_CONTEXT_VERSION = "morgan_query_region_coverage.v1"
MMP_LEDGER_CONTEXT_VERSION = "morgan_coverage_mmp_ledger.v2"


def attach_neighbor_context(
    retrieval: dict[str, Any],
    *,
    profile: str = STANDARD_NEIGHBOR_CONTEXT,
) -> dict[str, Any]:
    """Return an LLM-facing retrieval copy with optional coverage diagnostics."""
    _validate_profile(profile)
    if profile == STANDARD_NEIGHBOR_CONTEXT:
        return retrieval
    if not any(group.get("neighbors") for group in retrieval.get("groups") or []):
        return retrieval

    query = retrieval.get("query") or {}
    query_smiles = str(query.get("canonical_smiles") or query.get("input_smiles") or "")
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        raise ValueError("Cannot build coverage-aware context for an invalid query structure")

    fingerprint = query.get("fingerprint") or {}
    _validate_morgan_fingerprint(fingerprint)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(fingerprint.get("radius", 2)),
        fpSize=int(fingerprint.get("n_bits", 2048)),
        includeChirality=bool(fingerprint.get("useChirality", False)),
        useBondTypes=bool(fingerprint.get("useBondTypes", True)),
        includeRingMembership=True,
    )
    query_bits, query_bit_atoms = _fingerprint_with_atom_mapping(query_mol, generator)
    radius = int(fingerprint.get("radius", 2))

    output = deepcopy(retrieval)
    for group in output.get("groups") or []:
        neighbors = group.get("neighbors") or []
        if not neighbors:
            continue
        group["neighbor_set_context"] = _group_context(
            query_mol,
            query_bits,
            query_bit_atoms,
            neighbors,
            generator,
            radius,
        )
        if profile == COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT:
            context = group["neighbor_set_context"]
            _remove_approximate_atom_coverage(context)
            context["profile"] = profile
            context["version"] = MMP_LEDGER_CONTEXT_VERSION
            context["coverage_statistics_version"] = COVERAGE_CONTEXT_VERSION
            context["definition"] = (
                f"sequential coverage of query Morgan radius-{radius} features; "
                "concrete region location comes only from pairwise MMP/MCS results"
            )
            context["ledger_scope"] = (
                "set-level integration of pairwise mmp_structure_compare results "
                "for all selected analogs in this group"
            )
    output.setdefault("experiment", {})["neighbor_context_profile"] = profile
    output["experiment"]["neighbor_context_version"] = (
        MMP_LEDGER_CONTEXT_VERSION
        if profile == COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT
        else COVERAGE_CONTEXT_VERSION
    )
    return output


def attach_mmp_coverage_ledger(
    retrieval: dict[str, Any],
    tool_service: Any,
) -> dict[str, Any]:
    """Prefetch existing pairwise MMP/MCS text and attach it by analog rank.

    This is deliberately a thin orchestration layer.  Molecular comparison
    remains owned by the resident ``mmp_structure_compare`` service tool; this
    function only de-duplicates pairs and assembles the set-level prompt input.
    """
    profile = str((retrieval.get("experiment") or {}).get("neighbor_context_profile") or "")
    if profile != COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT:
        return retrieval

    query = retrieval.get("query") or {}
    query_smiles = str(query.get("canonical_smiles") or query.get("input_smiles") or "")
    if not query_smiles:
        raise ValueError("coverage_mmp_ledger requires a visible query structure")

    pair_order: list[str] = []
    for group in retrieval.get("groups") or []:
        for neighbor in group.get("neighbors") or []:
            neighbor_smiles = str(neighbor.get("canonical_smiles") or "")
            if neighbor_smiles and neighbor_smiles not in pair_order:
                pair_order.append(neighbor_smiles)
    calls = [
        (
            "mmp_structure_compare",
            {
                "query_smiles": query_smiles,
                "reference_smiles": neighbor_smiles,
                "max_mmp_alternatives": 5,
                "mcs_timeout_s": 5,
            },
        )
        for neighbor_smiles in pair_order
    ]
    results = tool_service.invoke_many(calls)
    if len(results) != len(pair_order):
        raise ValueError(
            "coverage_mmp_ledger tool result count mismatch: "
            f"{len(results)} != {len(pair_order)}"
        )
    by_neighbor = dict(zip(pair_order, results, strict=True))

    output = deepcopy(retrieval)
    for group in output.get("groups") or []:
        context = group.get("neighbor_set_context") or {}
        entries_by_rank = {
            int(entry.get("rank")): entry
            for entry in context.get("neighbors") or []
            if entry.get("rank") is not None
        }
        comparison_ok = True
        for neighbor in group.get("neighbors") or []:
            rank = int(neighbor.get("rank"))
            entry = entries_by_rank.get(rank)
            if entry is None:
                raise ValueError(
                    f"coverage_mmp_ledger missing coverage entry for neighbor rank {rank}"
                )
            result = by_neighbor[str(neighbor.get("canonical_smiles") or "")]
            status = str(result.get("status") or "error")
            comparison_ok = comparison_ok and status == "ok"
            entry["pairwise_structure_comparison"] = {
                "tool": "mmp_structure_compare.v1",
                "status": status,
                "text": str(result.get("content") or ""),
                "warnings": [str(item) for item in result.get("warnings") or []],
                "errors": [str(item) for item in result.get("errors") or []],
            }
        context["pairwise_structure_comparisons_complete"] = comparison_ok
        context["location_semantics"] = (
            "Use the MMP shared constant and reference-to-query variable-fragment "
            "transformation to identify concrete regions. Marginal Morgan counts "
            "measure complementarity but do not locate a fragment or establish atom "
            "coverage by themselves."
        )
    output.setdefault("experiment", {})["coverage_mmp_ledger_tool_execution"] = (
        "harness_prefetched_mmp_structure_compare.v1"
    )
    return output


def augment_group_messages_with_neighbor_context(
    messages: list[dict[str, Any]],
    group: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append coverage-specific instructions only when the opt-in context exists."""
    context = group.get("neighbor_set_context")
    if not context:
        return messages
    profile = str(context.get("profile") or "")
    is_mmp_ledger = profile == COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT
    payload = {
        "task": (
            "Build and use a set-level structural coverage ledger for the group analysis."
            if is_mmp_ledger
            else "Use the analog-set coverage context in the group analysis."
        ),
        "neighbor_set_context": context,
        "instructions": (
            [
                "Match every ledger entry to the visible query and analog by rank.",
                "For each rank, use the prefetched mmp_structure_compare shared constant, MCS coverage, and reference-to-query transformation to state which concrete query region its evidence represents.",
                "Compare those pairwise regions across ranks: identify what each later analog adds beyond earlier analogs, what is redundant, and which query regions remain unsupported by any analog.",
                "Use marginal Morgan counts only as set-level complementarity statistics; do not use them alone to name or locate a fragment.",
                "Do not infer whole-query atom coverage from Morgan feature coverage. MCS query coverage is the available coarse size measure.",
                "If no matched-pair transformation identifies a concrete region, mark that region correspondence unresolved rather than inventing a fragment-level mapping.",
                "Do not call mmp_structure_compare again unless the prefetched comparison failed or is genuinely insufficient. Other tools remain available for property-transfer questions.",
                "When analog evidence is transferable, combine complementary evidence across represented query regions; otherwise explain the mismatch and downweight it.",
                "MCS/MMP overlap is structural correspondence, not proof that a fragment causes the property and not a label vote.",
                "Reflect the rank-by-rank ledger, redundant coverage, and uncovered regions in the existing reasoning_summary, key_evidence, and caveats fields; keep the required JSON schema unchanged.",
            ]
            if is_mmp_ledger
            else [
                "Match coverage entries to analogs by rank.",
                "Distinguish complementary analogs that add coverage from redundant analogs that cover regions already represented.",
                "When analog evidence is transferable, combine complementary evidence across the covered query regions; when it is not transferable, say why and downweight it.",
                "Coverage means structural Morgan-environment overlap only. It is not proof that a region causes the measured property, and it is not a label vote.",
                "Reflect the coverage-aware integration in the existing reasoning_summary and evidence fields; keep the required JSON schema unchanged.",
            ]
        ),
    }
    return [
        *messages,
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def _group_context(
    query_mol: Chem.Mol,
    query_bits: frozenset[int],
    query_bit_atoms: dict[int, frozenset[int]],
    neighbors: list[dict[str, Any]],
    generator: Any,
    radius: int,
) -> dict[str, Any]:
    covered_bits: set[int] = set()
    covered_atoms: set[int] = set()
    entries: list[dict[str, Any]] = []
    for position, neighbor in enumerate(neighbors, start=1):
        neighbor_smiles = str(neighbor.get("canonical_smiles") or "")
        neighbor_mol = Chem.MolFromSmiles(neighbor_smiles)
        if neighbor_mol is None:
            raise ValueError(
                f"Cannot build coverage-aware context for neighbor rank {neighbor.get('rank', position)}"
            )
        neighbor_fp = generator.GetFingerprint(neighbor_mol)
        shared_bits = query_bits.intersection(int(bit) for bit in neighbor_fp.GetOnBits())
        marginal_bits = shared_bits - covered_bits
        redundant_bits = shared_bits & covered_bits
        shared_atoms = _atoms_for_bits(shared_bits, query_bit_atoms)
        marginal_atoms = _atoms_for_bits(marginal_bits, query_bit_atoms) - covered_atoms
        covered_bits.update(shared_bits)
        covered_atoms.update(shared_atoms)
        role = (
            "primary"
            if position == 1
            else "complementary"
            if marginal_bits or marginal_atoms
            else "redundant"
        )
        entries.append(
            {
                "rank": int(neighbor.get("rank", position)),
                "similarity": round(float(neighbor.get("similarity", 0.0)), 4),
                "coverage_role": role,
                "shared_query_feature_count": len(shared_bits),
                "marginal_new_query_feature_count": len(marginal_bits),
                "redundant_shared_query_feature_count": len(redundant_bits),
                "cumulative_query_feature_coverage": _fraction(len(covered_bits), len(query_bits)),
                "shared_query_atom_count": len(shared_atoms),
                "marginal_new_query_atom_count": len(marginal_atoms),
                "marginal_region_sizes": _component_sizes(query_mol, marginal_atoms),
                "cumulative_query_atom_coverage": _fraction(
                    len(covered_atoms), query_mol.GetNumAtoms()
                ),
            }
        )

    return {
        "profile": COVERAGE_AWARE_NEIGHBOR_CONTEXT,
        "version": COVERAGE_CONTEXT_VERSION,
        "definition": (
            f"sequential coverage of query Morgan radius-{radius} local environments "
            "and their mapped query atoms"
        ),
        "query_feature_count": len(query_bits),
        "query_atom_count": query_mol.GetNumAtoms(),
        "selected_set_query_feature_coverage": _fraction(len(covered_bits), len(query_bits)),
        "selected_set_query_atom_coverage": _fraction(
            len(covered_atoms), query_mol.GetNumAtoms()
        ),
        "neighbors": entries,
        "limitations": [
            "Morgan features are hashed local environments and can collide.",
            "Atom coverage is derived from the query environments represented by shared bits; it is not an MCS atom-to-atom alignment.",
            "Coverage does not establish a causal fragment-property contribution.",
        ],
    }


def _fingerprint_with_atom_mapping(
    mol: Chem.Mol,
    generator: Any,
) -> tuple[frozenset[int], dict[int, frozenset[int]]]:
    additional = rdFingerprintGenerator.AdditionalOutput()
    additional.AllocateBitInfoMap()
    fingerprint = generator.GetFingerprint(mol, additionalOutput=additional)
    mapping: dict[int, frozenset[int]] = {}
    for raw_bit, environments in additional.GetBitInfoMap().items():
        atoms: set[int] = set()
        for center, radius in environments:
            atoms.update(_environment_atoms(mol, int(center), int(radius)))
        mapping[int(raw_bit)] = frozenset(atoms)
    return frozenset(int(bit) for bit in fingerprint.GetOnBits()), mapping


def _environment_atoms(mol: Chem.Mol, center: int, radius: int) -> set[int]:
    atoms = {center}
    if radius <= 0:
        return atoms
    for bond_index in Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center):
        bond = mol.GetBondWithIdx(int(bond_index))
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    return atoms


def _atoms_for_bits(
    bits: set[int] | frozenset[int],
    bit_atoms: dict[int, frozenset[int]],
) -> set[int]:
    atoms: set[int] = set()
    for bit in bits:
        atoms.update(bit_atoms.get(bit, ()))
    return atoms


def _component_sizes(mol: Chem.Mol, atom_indices: set[int]) -> list[int]:
    remaining = set(atom_indices)
    sizes: list[int] = []
    while remaining:
        frontier = [remaining.pop()]
        size = 0
        while frontier:
            atom_index = frontier.pop()
            size += 1
            for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if neighbor_index in remaining:
                    remaining.remove(neighbor_index)
                    frontier.append(neighbor_index)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _remove_approximate_atom_coverage(context: dict[str, Any]) -> None:
    """Remove the permissive v1 atom proxy from the MMP-ledger experiment.

    Folded Morgan feature overlap can touch every query atom even when the MCS
    is small.  The legacy coverage-aware profile retains those diagnostics for
    provenance, while the new ledger relies on MCS/MMP for region semantics.
    """
    for field in ("query_atom_count", "selected_set_query_atom_coverage"):
        context.pop(field, None)
    for entry in context.get("neighbors") or []:
        for field in (
            "shared_query_atom_count",
            "marginal_new_query_atom_count",
            "marginal_region_sizes",
            "cumulative_query_atom_coverage",
        ):
            entry.pop(field, None)


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _validate_profile(profile: str) -> None:
    if profile not in NEIGHBOR_CONTEXT_PROFILES:
        raise ValueError(
            f"Unknown neighbor context profile {profile!r}; expected one of "
            + ", ".join(NEIGHBOR_CONTEXT_PROFILES)
        )


def _validate_morgan_fingerprint(fingerprint: dict[str, Any]) -> None:
    fingerprint_type = str(fingerprint.get("type") or "")
    if "morgan" not in fingerprint_type.lower():
        raise ValueError(
            "coverage_aware neighbor context currently requires an RDKit Morgan retrieval fingerprint"
        )
    if bool(fingerprint.get("useFeatures", False)):
        raise ValueError(
            "coverage_aware neighbor context does not yet support feature-invariant Morgan fingerprints"
        )
