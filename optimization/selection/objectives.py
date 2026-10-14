"""Shared molecular coverage, diversity and greedy context objectives."""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import heapq
import math
from typing import Any, Iterable, Mapping, Sequence
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

OBJECTIVE_VERSION = "morgan_assay_feature_semantic.v1"
GATED_OBJECTIVE_VERSION = "morgan_assay_gated_feature_semantic.v1"
NORMALIZED_GATED_OBJECTIVE_VERSION = "morgan_normalized_gated_feature_semantic.v1"
DIRECT_BUDGET = 10
INDIRECT_BUDGET = 50
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_DIRECT_MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=True
)


def _finite_unit(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return number


@lru_cache(maxsize=None)
def _fingerprint(
    smiles: str, *, chirality: bool = False
) -> DataStructs.ExplicitBitVect:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid cached parent SMILES: {smiles!r}")
    return (_DIRECT_MORGAN if chirality else _MORGAN).GetFingerprint(molecule)


@dataclass(frozen=True)
class DirectProfile:
    name: str
    gated_assay: float
    molecular: float
    label: float


@dataclass(frozen=True)
class IndirectProfile:
    name: str
    gated_assay: float
    molecular: float
    semantic_relevance: float
    semantic_diversity: float
    level_diversity: float
    aliases: tuple[str, ...]
    record_diversity: float = 0.0
    dili_geomean: bool = False


@dataclass(frozen=True)
class PreparedCandidates:
    rows: tuple[dict[str, Any], ...]
    bits: tuple[frozenset[int], ...]
    universe: frozenset[int]
    molecular_coverage_ceiling: float


def _log_ceiling(capacities: Mapping[str, int], budget: int) -> float:
    return _group_ceiling(capacities, budget, "log")


def _group_gain(count: int, function: str) -> float:
    if function == "sqrt":
        return math.sqrt(count + 1) - math.sqrt(count)
    if function == "log":
        return math.log(count + 2) - math.log(count + 1)
    raise ValueError(f"Unknown diversity function: {function}")


def _group_ceiling(capacities: Mapping[str, int], budget: int, function: str) -> float:
    if budget < 1 or sum(capacities.values()) < budget:
        raise ValueError("Group capacities cannot fill the requested budget")
    counts = {group: 0 for group in capacities}
    heap = [
        (-_group_gain(0, function), group) for group, cap in capacities.items() if cap
    ]
    heapq.heapify(heap)
    total = 0.0
    for _ in range(budget):
        if not heap:
            raise ValueError("Group capacities cannot fill the requested budget")
        negative_gain, group = heapq.heappop(heap)
        total -= negative_gain
        counts[group] += 1
        if counts[group] < capacities[group]:
            gain = _group_gain(counts[group], function)
            heapq.heappush(heap, (-gain, group))
    return total


def normalized_log_diversity(
    selected_groups: Iterable[str], capacities: Mapping[str, int], budget: int
) -> float:
    counts = Counter(map(str, selected_groups))
    if any((counts[group] > capacities.get(group, 0) for group in counts)):
        raise ValueError("Selected group count exceeds candidate capacity")
    denominator = _log_ceiling(capacities, budget)
    value = sum((math.log1p(count) for count in counts.values())) / denominator
    if not -1e-12 <= value <= 1 + 1e-12:
        raise AssertionError(f"Normalized log diversity escaped [0, 1]: {value}")
    return min(1.0, max(0.0, value))


def normalized_sqrt_diversity(
    selected_groups: Iterable[str], capacities: Mapping[str, int], budget: int
) -> float:
    counts = Counter(map(str, selected_groups))
    if any((counts[group] > capacities.get(group, 0) for group in counts)):
        raise ValueError("Selected group count exceeds candidate capacity")
    value = sum((math.sqrt(count) for count in counts.values())) / _group_ceiling(
        capacities, budget, "sqrt"
    )
    if not -1e-12 <= value <= 1 + 1e-12:
        raise AssertionError(f"Normalized sqrt diversity escaped [0, 1]: {value}")
    return min(1.0, max(0.0, value))


def _validate_rows(
    rows: Sequence[Mapping[str, Any]], *, direct: bool
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    ids = [str(row["item_id"]) for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate identifiers must be unique")
    for row in output:
        for field in ("morgan_similarity", "assay_transfer_score"):
            value = row.get(field)
            if value is None and field == "assay_transfer_score":
                continue
            number = float(value)
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError(f"{field} must be finite and within [0, 1]")
            row[field] = number
        if direct and int(row["gold_label"]) not in (0, 1):
            raise ValueError("Canonical direct card labels must be binary")
        if not direct:
            if not str(row.get("level") or ""):
                raise ValueError("Indirect candidates require a level")
            if (
                not row.get("semantic_bucket_id")
                and row.get("semantic_weight_source") != "provisional_task_level_mean"
            ):
                raise ValueError(
                    "Indirect candidates require a reviewed semantic bucket or provisional mean"
                )
            weight = row.get("semantic_weight")
            if weight is not None and (
                not math.isfinite(float(weight)) or not 0 <= float(weight) <= 1
            ):
                raise ValueError("Semantic weights must be within [0, 1]")
            tool_score = row.get("dili_tool_score")
            if tool_score is not None and (
                not math.isfinite(float(tool_score)) or not 0 <= float(tool_score) <= 1
            ):
                raise ValueError("DILI tool scores must be within [0, 1]")
    return output


def _prepare_candidates(
    candidates: Sequence[Mapping[str, Any]], *, direct: bool, budget: int
) -> PreparedCandidates:
    rows = _validate_rows(candidates, direct=direct)
    if budget < 1 or budget > len(rows):
        raise ValueError(
            "Budget must be positive and no greater than the candidate count"
        )
    bits = tuple(
        (
            frozenset(
                _fingerprint(
                    str(row["canonical_smiles"] if direct else row["parent_smiles"]),
                    chirality=direct,
                ).GetOnBits()
            )
            for row in rows
        )
    )
    universe = frozenset((bit for current in bits for bit in current))
    return PreparedCandidates(tuple(rows), bits, universe, 1.0)


def _lazy_select(
    prepared: PreparedCandidates,
    *,
    budget: int,
    base_scores: Sequence[float],
    molecular_lambda: float,
    group_terms: Sequence[tuple[float, Sequence[str], float]],
    diversity_function: str = "log",
) -> tuple[list[int], list[float], float, float]:
    rows, bits, universe = (prepared.rows, prepared.bits, prepared.universe)
    raw_ceiling = prepared.molecular_coverage_ceiling
    coverage_ceiling = raw_ceiling or 1.0
    counts = [Counter() for _ in group_terms]
    covered: set[int] = set()
    chosen: list[int] = []
    gains: list[float] = []
    remaining = set(range(len(rows)))

    def marginal(index: int) -> float:
        gain = float(base_scores[index]) / budget
        if molecular_lambda:
            gain += molecular_lambda * (
                len(bits[index] - covered) / len(universe) / coverage_ceiling
                if universe
                else 0.0
            )
        for term_index, (weight, groups, denominator) in enumerate(group_terms):
            if weight:
                count = counts[term_index][groups[index]]
                gain += weight * _group_gain(count, diversity_function) / denominator
        return gain

    heap = [
        (-marginal(index), str(rows[index]["item_id"]), index) for index in remaining
    ]
    heapq.heapify(heap)
    while len(chosen) < budget:
        _, tie, index = heapq.heappop(heap)
        if index not in remaining:
            continue
        gain = marginal(index)
        next_bound = -heap[0][0] if heap else -math.inf
        if gain + 1e-15 < next_bound:
            heapq.heappush(heap, (-gain, tie, index))
            continue
        chosen.append(index)
        gains.append(gain)
        remaining.remove(index)
        covered.update(bits[index])
        for term_index, (_, groups, _) in enumerate(group_terms):
            counts[term_index][groups[index]] += 1
    coverage = (
        min(1.0, len(covered) / len(universe) / coverage_ceiling) if universe else 0.0
    )
    return (chosen, gains, coverage, raw_ceiling)


def select_direct_contexts(
    candidates: Sequence[Mapping[str, Any]],
    *,
    profile: DirectProfile,
    budget: int = DIRECT_BUDGET,
    prepared: PreparedCandidates | None = None,
    assay_enabled: bool = True,
    sqrt_gated_assay: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = prepared or _prepare_candidates(candidates, direct=True, budget=budget)
    rows = prepared.rows
    labels = [str(int(row["gold_label"])) for row in rows]
    capacities = Counter(labels)
    label_ceiling = _log_ceiling(capacities, budget)
    base = [
        row["morgan_similarity"]
        if not assay_enabled or row.get("assay_transfer_score") is None
        else (
            row["morgan_similarity"]
            + profile.gated_assay
            * (
                math.sqrt(row["morgan_similarity"] * row["assay_transfer_score"])
                if sqrt_gated_assay
                else row["morgan_similarity"] * row["assay_transfer_score"]
            )
        )
        / (1 + profile.gated_assay)
        for row in rows
    ]
    ranking, gains, molecular, coverage_ceiling = _lazy_select(
        prepared,
        budget=budget,
        base_scores=base,
        molecular_lambda=profile.molecular,
        group_terms=((profile.label, labels, label_ceiling),),
    )
    selected = [
        {**rows[index], "selection_rank": rank, "marginal_gain": gains[rank - 1]}
        for rank, index in enumerate(ranking, start=1)
    ]
    return (
        selected,
        {
            "normalized_relevance": sum((base[index] for index in ranking)) / budget,
            "molecular_coverage_normalized": molecular,
            "molecular_coverage_ceiling": coverage_ceiling,
            "label_diversity": normalized_log_diversity(
                (labels[index] for index in ranking), capacities, budget
            ),
            "label_0": sum((labels[index] == "0" for index in ranking)),
            "label_1": sum((labels[index] == "1" for index in ranking)),
            "candidate_contexts": len(rows),
            "selected_contexts": budget,
            "objective_score": sum(gains),
        },
    )


def select_joint_indirect(
    candidates: Sequence[Mapping[str, Any]],
    *,
    profile: IndirectProfile,
    budget: int = INDIRECT_BUDGET,
    prepared: PreparedCandidates | None = None,
    diversity_function: str = "log",
    assay_enabled: bool = True,
    sqrt_gated_assay: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = prepared or _prepare_candidates(candidates, direct=False, budget=budget)
    rows = prepared.rows
    levels = [str(row["level"]) for row in rows]
    semantics = [str(row["semantic_bucket_id"] or "__unassigned__") for row in rows]
    level_capacities, semantic_capacities = (Counter(levels), Counter(semantics))
    level_ceiling = _group_ceiling(level_capacities, budget, diversity_function)
    semantic_ceiling = _group_ceiling(semantic_capacities, budget, diversity_function)
    if profile.record_diversity and diversity_function != "sqrt":
        raise ValueError("Record diversity requires square-root group gains")
    parents = (
        [str(row.get("parent_id") or "") for row in rows]
        if profile.record_diversity
        else []
    )
    if profile.record_diversity and (not all(parents)):
        raise ValueError("Record diversity requires every candidate parent_id")
    parent_capacities = Counter(parents)
    parent_ceiling = (
        _group_ceiling(parent_capacities, budget, "sqrt") if parents else 1.0
    )
    base, semantic_weights = ([], [])
    for row in rows:
        similarity = float(row["morgan_similarity"])
        assay = row.get("assay_transfer_score")
        base.append(
            similarity
            if not assay_enabled or assay is None
            else (
                similarity
                + profile.gated_assay
                * (
                    math.sqrt(similarity * float(assay))
                    if sqrt_gated_assay
                    else similarity * float(assay)
                )
            )
            / (1 + profile.gated_assay)
        )
        semantic_weights.append(float(row.get("semantic_weight") or 0.0))
    semantic_available = all((row.get("semantic_weight") is not None for row in rows))
    if profile.dili_geomean and (not semantic_available):
        raise ValueError("DILI geometric relevance requires every semantic weight")
    sr_effective = profile.semantic_relevance if semantic_available else 0.0
    geometric = (
        [
            math.sqrt(float(row["dili_tool_score"]) * semantic_weights[index])
            for index, row in enumerate(rows)
        ]
        if profile.dili_geomean
        else []
    )
    modular = [
        base[index]
        + sr_effective
        * (geometric[index] if profile.dili_geomean else semantic_weights[index])
        for index in range(len(rows))
    ]
    ranking, gains, molecular, coverage_ceiling = _lazy_select(
        prepared,
        budget=budget,
        base_scores=modular,
        molecular_lambda=profile.molecular,
        group_terms=(
            (profile.semantic_diversity, semantics, semantic_ceiling),
            (profile.level_diversity, levels, level_ceiling),
        )
        + (((profile.record_diversity, parents, parent_ceiling),) if parents else ()),
        diversity_function=diversity_function,
    )
    selected = [
        {**rows[index], "selection_rank": rank, "marginal_gain": gains[rank - 1]}
        for rank, index in enumerate(ranking, start=1)
    ]
    selected_levels = [levels[index] for index in ranking]
    normalize = (
        normalized_sqrt_diversity
        if diversity_function == "sqrt"
        else normalized_log_diversity
    )
    return (
        selected,
        {
            "normalized_relevance": sum((base[index] for index in ranking)) / budget,
            "molecular_coverage_normalized": molecular,
            "molecular_coverage_ceiling": coverage_ceiling,
            "semantic_relevance": sum((semantic_weights[index] for index in ranking))
            / budget,
            **(
                {
                    "dili_geometric_relevance": sum(
                        (geometric[index] for index in ranking)
                    )
                    / budget,
                    "dili_tool_score": sum(
                        (float(rows[index]["dili_tool_score"]) for index in ranking)
                    )
                    / budget,
                }
                if profile.dili_geomean
                else {}
            ),
            "semantic_relevance_available": semantic_available,
            "semantic_relevance_lambda_effective": sr_effective,
            "semantic_diversity": normalize(
                (semantics[index] for index in ranking), semantic_capacities, budget
            ),
            "level_diversity": normalize(selected_levels, level_capacities, budget),
            **(
                {
                    "record_diversity": normalized_sqrt_diversity(
                        (parents[index] for index in ranking), parent_capacities, budget
                    ),
                    "selected_parent_count": len({parents[index] for index in ranking}),
                }
                if parents
                else {}
            ),
            "level_counts": dict(sorted(Counter(selected_levels).items())),
            "candidate_records": len(rows),
            "selected_records": budget,
            "objective_score": sum(gains),
        },
    )
