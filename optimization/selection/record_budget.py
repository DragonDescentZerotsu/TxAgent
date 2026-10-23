"""Deterministic raw-record controls used by the paper baselines."""
import hashlib

VERSION = "budgeted_evidence.v2"
SIMPLE_POLICIES = ("neighbor_fill",)


def select_simple_records(records, *, budget, policy='neighbor_fill', per_molecule=5, seed=0):
    """Paper Neighbor-fill; preserve frozen seed/order and audit schema."""
    if budget < 0 or per_molecule < 1:
        raise ValueError('invalid record budget or molecule cap')
    records = list(records)
    if policy not in SIMPLE_POLICIES and not (policy == "molecule_cap" and not records):
        raise ValueError("paper baselines require neighbor_fill")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate candidate IDs")
    if any(not r["molecule_id"] or not r["group"] for r in records):
        raise ValueError("each record requires one molecule and one group")
    ordered = sorted(records, key=lambda r: (-r.get("similarity", 0), r["id"]))
    mol_order = {}
    for r in ordered:
        mol_order.setdefault(r["molecule_id"], len(mol_order))
    ordered.sort(
        key=lambda r: (
            mol_order[r["molecule_id"]],
            hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest(),
        )
    )
    selected = ordered[:budget]
    return selected, {
        "version": VERSION,
        "policy": policy,
        "budget": budget,
        "per_molecule": None if policy == "neighbor_fill" else per_molecule,
        "seed": seed,
        "n_candidates": len(records),
        "n_selected": len(selected),
        "shortfall": budget - len(selected),
        "n_molecules": len({r["molecule_id"] for r in selected}),
        "n_groups": len({r["group"] for r in selected}),
        "assessment_required": False,
        "selection_trace": [{"record_id": r["id"]} for r in selected],
    }
