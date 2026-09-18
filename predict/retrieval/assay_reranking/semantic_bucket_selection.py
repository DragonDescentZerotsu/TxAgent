"""Deterministic semantic- and pair-bucket-balanced record selection."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


MINIMUM_RELEVANCE_PERCENTILE = 20.0


def select_molecule_records(
    rows: Sequence[Mapping[str, Any]], *, method: str, limit: int = 10,
) -> list[dict[str, Any]]:
    """Select eligible records within one Morgan-ranked parent molecule."""
    if method not in {"control", "semantic"} or limit < 1:
        raise ValueError("method must be control or semantic with a positive limit")
    selected = []
    for source in rows:
        row = dict(source)
        if not isinstance(row.get("retrieval_eligible"), bool):
            raise ValueError("within-molecule rows require binary retrieval eligibility")
        if not row["retrieval_eligible"]:
            continue
        if not row.get("tie_key"):
            raise ValueError("within-molecule rows require a stable tie key")
        if method == "semantic" and row.get("semantic_rank") is None:
            raise ValueError("semantic selection requires a bucket rank")
        selected.append(row)
    key = (
        (lambda row: (int(row["semantic_rank"]), str(row["semantic_bucket_id"]), row["tie_key"]))
        if method == "semantic" else (lambda row: row["tie_key"])
    )
    return sorted(selected, key=key)[:limit]


def select_global_molecule_records(
    rows: Sequence[Mapping[str, Any]], *, method: str, limit: int = 25,
    per_parent_limit: int = 10, per_parent_bucket_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select a globally capped record panel from fixed Morgan parents."""
    if method not in {"control", "semantic"} or min(limit, per_parent_limit) < 1:
        raise ValueError("method and positive total/per-parent limits are required")
    if per_parent_bucket_limit is not None and per_parent_bucket_limit < 1:
        raise ValueError("per-parent semantic-bucket limit must be positive")
    ranked = []
    for source in rows:
        row = dict(source)
        required = ("parent_id", "parent_rank", "tie_key")
        if any(row.get(field) in {None, ""} for field in required):
            raise ValueError("global molecule rows require parent identity, rank, and tie key")
        if method == "semantic":
            if row.get("retrieval_eligible") is not True or row.get("semantic_rank") is None:
                continue
            if not row.get("semantic_bucket_id"):
                raise ValueError("eligible semantic rows require a bucket identity")
        ranked.append(row)
    if method == "semantic":
        ranked.sort(key=lambda row: (
            int(row["semantic_rank"]), int(row["parent_rank"]),
            str(row["semantic_bucket_id"]), str(row["tie_key"]),
        ))
    else:
        ranked.sort(key=lambda row: str(row["tie_key"]))
    selected, counts, bucket_counts = [], defaultdict(int), defaultdict(int)
    for row in ranked:
        parent = str(row["parent_id"])
        if counts[parent] >= per_parent_limit:
            continue
        cell = (parent, str(row.get("semantic_bucket_id") or ""))
        if (method == "semantic" and per_parent_bucket_limit is not None
                and bucket_counts[cell] >= per_parent_bucket_limit):
            continue
        selected.append(row)
        counts[parent] += 1
        bucket_counts[cell] += 1
        if len(selected) == limit:
            break
    return selected


def select_semantic_lap(
    rows: Sequence[Mapping[str, Any]], *, limit: int, per_bucket: int = 3,
) -> list[dict[str, Any]]:
    """Visit ranked semantic buckets, taking the next records on each lap."""
    if limit < 1 or per_bucket < 1:
        raise ValueError("limit and per_bucket must be positive")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ranks: dict[str, int] = {}
    for source in rows:
        row = dict(source)
        bucket = str(row.get("semantic_bucket_id") or "")
        rank = row.get("semantic_rank")
        similarity = row.get("morgan_similarity")
        tie_key = str(row.get("tie_key") or "")
        if not bucket or rank is None or similarity is None or not tie_key:
            raise ValueError("semantic rows require bucket, rank, similarity, and tie key")
        if bucket in ranks and ranks[bucket] != int(rank):
            raise ValueError(f"semantic bucket has conflicting ranks: {bucket}")
        ranks[bucket] = int(rank)
        row["morgan_similarity"] = float(similarity)
        buckets[bucket].append(row)
    for records in buckets.values():
        records.sort(key=lambda row: (-row["morgan_similarity"], row["tie_key"]))

    ordered = sorted(buckets, key=lambda bucket: (ranks[bucket], bucket))
    selected: list[dict[str, Any]] = []
    lap = 0
    while len(selected) < limit:
        added = False
        start = lap * per_bucket
        for bucket in ordered:
            for row in buckets[bucket][start:start + per_bucket]:
                item = dict(row, semantic_lap=lap + 1)
                selected.append(item)
                added = True
                if len(selected) == limit:
                    return selected
        if not added:
            break
        lap += 1
    return selected


def select_semantic_weighted(
    rows: Sequence[Mapping[str, Any]], *, limit: int, per_bucket_limit: int = 8,
) -> list[dict[str, Any]]:
    """Take the strongest expert-weighted Morgan records with one diversity cap."""
    if limit < 1 or per_bucket_limit < 1:
        raise ValueError("limit and per_bucket_limit must be positive")
    ranked = []
    for source in rows:
        row = dict(source)
        bucket = str(row.get("semantic_bucket_id") or "")
        similarity = row.get("morgan_similarity")
        weight = row.get("expert_weight")
        tie_key = str(row.get("tie_key") or "")
        if not bucket or similarity is None or weight is None or not tie_key:
            raise ValueError("weighted semantic rows require bucket, weight, similarity, and tie key")
        similarity, weight = float(similarity), float(weight)
        if not 0 <= similarity <= 1 or not 0 <= weight <= 1:
            raise ValueError("semantic weight and Morgan similarity must be between zero and one")
        if weight == 0:
            continue
        row.update(
            morgan_similarity=similarity,
            expert_weight=weight,
            semantic_utility=weight * similarity,
        )
        ranked.append(row)
    ranked.sort(key=lambda row: (-row["semantic_utility"], row["tie_key"]))
    selected, counts = [], defaultdict(int)
    for row in ranked:
        bucket = row["semantic_bucket_id"]
        if counts[bucket] == per_bucket_limit:
            continue
        selected.append(row)
        counts[bucket] += 1
        if len(selected) == limit:
            break
    return selected


def select_records(
    rows: Sequence[Mapping[str, Any]], *, method: str, limit: int,
) -> list[dict[str, Any]]:
    """Take one eligible record per pair bucket per pass, in priority order."""
    if method not in {"morgan", "assay_transfer"} or limit < 1:
        raise ValueError("method must be morgan or assay_transfer and limit must be positive")
    score_field = "morgan_similarity" if method == "morgan" else "assay_transfer_score"
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        percentile = float(row["semantic_percentile"])
        score = row.get(score_field)
        if percentile < MINIMUM_RELEVANCE_PERCENTILE:
            continue
        if score is None:
            raise ValueError(f"Eligible record lacks {score_field}: {row.get('record_id')}")
        pair_bucket = str(row.get("pair_bucket_key") or "")
        tie_key = str(row.get("tie_key") or "")
        if not pair_bucket or not tie_key:
            raise ValueError("Eligible records require pair_bucket_key and tie_key")
        row[score_field] = float(score)
        row["semantic_percentile"] = percentile
        buckets[pair_bucket].append(row)

    for records in buckets.values():
        records.sort(key=lambda row: (-row[score_field], row["tie_key"]))
    ordered_buckets = sorted(
        buckets,
        key=lambda key: (
            -max(row["semantic_percentile"] for row in buckets[key]),
            -buckets[key][0][score_field],
            key,
        ),
    )
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for key in ordered_buckets:
            if offset < len(buckets[key]):
                selected.append(buckets[key][offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected
