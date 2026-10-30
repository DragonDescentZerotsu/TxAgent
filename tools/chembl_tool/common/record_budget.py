"""Deterministic record-budget controls shared by retrieval and frozen-input ablations."""

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math

VERSION = "budgeted_evidence.v2"
DEFAULT_BUDGET = 64
SIMPLE_POLICIES = ("neighbor_fill", "molecule_cap", "group_balanced")
POLICIES = (*SIMPLE_POLICIES, "quality", "joint")


@dataclass(frozen=True)
class SelectionConfig:
    budget: int = DEFAULT_BUDGET
    policy: str = "joint"
    weights: tuple[float, float, float] = (0.5, 0.375, 0.125)
    per_molecule: int = 5
    seed: int = 0

    def __post_init__(self):
        if self.budget < 0 or self.per_molecule < 1 or self.policy not in POLICIES:
            raise ValueError("invalid selection budget, molecule cap or policy")
        if len(self.weights) != 3 or any(
            not math.isfinite(w) or w < 0 for w in self.weights
        ):
            raise ValueError("weights must be three finite nonnegative numbers")
        if not math.isclose(sum(self.weights), 1, abs_tol=1e-9):
            raise ValueError("weights must sum to one")


def select_simple_records(records, config):
    """Raw-card controls: no assessed features or quality scores are needed."""
    records = list(records)
    if config.policy not in SIMPLE_POLICIES:
        raise ValueError("simple selection requires a simple policy")
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
            hashlib.sha256(f"{config.seed}:{r['id']}".encode()).hexdigest(),
        )
    )
    pools = defaultdict(deque)
    for r in ordered:
        pools[r["group"] if config.policy == "group_balanced" else "all"].append(r)
    counts, selected = defaultdict(int), []
    while pools and len(selected) < config.budget:
        for group in sorted(list(pools)):
            pool = pools[group]
            # Skip capped molecules within this group's turn, so every nonempty
            # eligible group contributes once per round, not once per raw offset.
            while pool:
                r = pool.popleft()
                if (
                    config.policy != "neighbor_fill"
                    and counts[r["molecule_id"]] >= config.per_molecule
                ):
                    continue
                counts[r["molecule_id"]] += 1
                selected.append(r)
                break
            if not pool:
                del pools[group]
            if len(selected) == config.budget:
                break
    return selected, {
        "version": VERSION,
        "policy": config.policy,
        "budget": config.budget,
        "per_molecule": None
        if config.policy == "neighbor_fill"
        else config.per_molecule,
        "seed": config.seed,
        "n_candidates": len(records),
        "n_selected": len(selected),
        "shortfall": config.budget - len(selected),
        "n_molecules": len(counts),
        "n_groups": len({r["group"] for r in selected}),
        "assessment_required": False,
        "selection_trace": [{"record_id": r["id"]} for r in selected],
    }


def content_hash(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
