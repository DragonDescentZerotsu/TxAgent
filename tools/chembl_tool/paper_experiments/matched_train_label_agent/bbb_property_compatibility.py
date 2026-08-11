"""Pure, label-blind contracts for BBB property-compatible neighbor selection."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


FEATURE_NAMES = (
    "pka__fraction_neutral",
    "pka__logd_estimate",
    "rdkit__MolWt",
    "rdkit__TPSA",
    "rdkit__NumHDonors",
    "rdkit__NumRotatableBonds",
)
ION_FEATURES = (
    "pka__fraction_neutral",
    "pka__num_acidic_sites",
    "pka__num_basic_sites",
)


def ionization_class(features: dict[str, float | None]) -> str:
    neutral = features.get("pka__fraction_neutral")
    acidic = features.get("pka__num_acidic_sites")
    basic = features.get("pka__num_basic_sites")
    if neutral is None or acidic is None or basic is None:
        return "unknown"
    if float(neutral) >= 0.5:
        return "mostly_neutral"
    has_acid, has_base = float(acidic) > 0, float(basic) > 0
    if has_acid and has_base:
        return "mostly_ionized_ampholytic"
    if has_acid:
        return "mostly_ionized_acidic"
    if has_base:
        return "mostly_ionized_basic"
    return "mostly_ionized_other"


def robust_feature_scales(
    rows: Iterable[dict[str, float | None]],
) -> dict[str, dict[str, float]]:
    materialized = list(rows)
    result = {}
    for name in FEATURE_NAMES:
        values = sorted(
            float(row[name])
            for row in materialized
            if row.get(name) is not None and math.isfinite(float(row[name]))
        )
        if not values:
            raise ValueError(f"No finite train values for {name}")
        median = statistics.median(values)
        scale = _percentile(values, 0.75) - _percentile(values, 0.25)
        if scale <= 1e-12:
            scale = max(statistics.median(abs(value - median) for value in values), 1.0)
        result[name] = {"median": median, "scale": scale, "n": len(values)}
    return result


def property_distance(
    left: dict[str, float | None],
    right: dict[str, float | None],
    scales: dict[str, dict[str, float]],
) -> float:
    distances = [
        abs(float(left[name]) - float(right[name])) / scales[name]["scale"]
        for name in FEATURE_NAMES
        if left.get(name) is not None and right.get(name) is not None
    ]
    return sum(distances) / len(distances) if distances else math.inf


def select_compatible_neighbors(
    candidates: list[dict[str, Any]],
    *,
    query_features: dict[str, float | None],
    train_features: list[dict[str, float | None]],
    scales: dict[str, dict[str, float]],
    top_k: int,
    similarity_floor: float,
) -> list[dict[str, Any]]:
    eligible = decorate_candidates(candidates, query_features, train_features, scales)
    eligible = [
        row for row in eligible if float(row["similarity"]) + 1e-12 >= similarity_floor
    ]
    if len(eligible) < top_k:
        raise ValueError(f"Only {len(eligible)} candidates satisfy the similarity floor")
    return sorted(
        eligible,
        key=lambda row: (
            row["ionization_mismatch"],
            row["property_distance"],
            -float(row["similarity"]),
            int(row["train_index"]),
        ),
    )[:top_k]


def decorate_candidates(
    candidates: list[dict[str, Any]],
    query: dict[str, float | None],
    train: list[dict[str, float | None]],
    scales: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    query_class = ionization_class(query)
    result = []
    for candidate in candidates:
        features = train[int(candidate["train_index"])]
        candidate_class = ionization_class(features)
        result.append(
            {
                **candidate,
                "ionization_class": candidate_class,
                "ionization_mismatch": int(
                    query_class == "unknown"
                    or candidate_class == "unknown"
                    or query_class != candidate_class
                ),
                "property_distance": property_distance(query, features, scales),
            }
        )
    return result


def selection_summary(neighbors: list[dict[str, Any]]) -> dict[str, Any]:
    score = sum(int(row["Y"]) for row in neighbors) / len(neighbors)
    return {
        "train_indices": [int(row["train_index"]) for row in neighbors],
        "morgan_ranks": [int(row["morgan_rank"]) for row in neighbors],
        "labels": [int(row["Y"]) for row in neighbors],
        "prediction": int(score >= 0.5),
        "mean_similarity": statistics.fmean(float(row["similarity"]) for row in neighbors),
        "mean_property_distance": statistics.fmean(
            float(row["property_distance"]) for row in neighbors
        ),
        "same_ionization_count": sum(not row["ionization_mismatch"] for row in neighbors),
    }


def morgan_top_n(
    train: list[dict[str, Any]], valid: list[dict[str, Any]], *, top_n: int
) -> list[list[dict[str, Any]]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False, useBondTypes=True
    )
    train_fps = [_fingerprint(row["drug"], generator) for row in train]
    result = []
    for query in valid:
        similarities = DataStructs.BulkTanimotoSimilarity(
            _fingerprint(query["drug"], generator), train_fps
        )
        ranked = sorted(
            range(len(train)), key=lambda index: (-similarities[index], f"{index:012d}")
        )[:top_n]
        result.append(
            [
                {
                    "train_index": index,
                    "Y": int(train[index]["Y"]),
                    "similarity": float(similarities[index]),
                    "morgan_rank": rank,
                }
                for rank, index in enumerate(ranked, start=1)
            ]
        )
    return result


def budget_key(value: float) -> str:
    return f"similarity_cost_le_{value:.2f}"


def _fingerprint(
    smiles: str, generator: rdFingerprintGenerator.FingerprintGenerator64
) -> DataStructs.ExplicitBitVect:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return generator.GetFingerprint(molecule)


def _percentile(values: list[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight
