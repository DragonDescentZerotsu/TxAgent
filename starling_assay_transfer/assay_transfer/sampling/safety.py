from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any, Mapping, Iterable
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from assay_transfer.targets import stable_sha256
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def _selected_edges(train: list[dict], calibration: Mapping[str, Any], fps: Mapping[str, Any],
                    limit: int | None, allow_shortfall: bool = False,
                    cap: int = 96) -> tuple[np.ndarray, list[dict], int]:
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(train):
        buckets[row["pair_bucket_key"]][row["parent_id"]].append(index)
    parent_orders = {bucket: sorted(parents, key=lambda parent: stable_sha256(
        f"v28:parent:{bucket}:{parent}")) for bucket, parents in buckets.items()}
    schedule = sorted(range(len(train)), key=lambda index: stable_sha256(
        f"v28:query:{train[index]['record_id']}"))
    usage, seen, edges = Counter(), defaultdict(set), []
    for round_index in range(cap):
        for query_index in schedule:
            result = _fast_reference(query_index, round_index, train, calibration, fps,
                                     buckets, parent_orders, usage, seen, cap)
            if result is None:
                continue
            reference_index, near = result
            usage[reference_index] += 1
            seen[query_index].add(train[reference_index]["parent_id"])
            edges.append((query_index, reference_index, int(near)))
            if len(edges) == limit:
                counts = Counter((train[q]["measurement_kind"], bool(near))
                                 for q, _, near in edges)
                phases = [{"family": family, "pair_type": "near" if near else "far",
                           "pairs": count} for (family, near), count in sorted(counts.items())]
                return np.asarray(edges, dtype=np.int32), phases, cap
    if (allow_shortfall or limit is None) and edges:
        counts = Counter((train[q]["measurement_kind"], bool(near)) for q, _, near in edges)
        phases = [{"family": family, "pair_type": "near" if near else "far", "pairs": count}
                  for (family, near), count in sorted(counts.items())]
        return np.asarray(edges, dtype=np.int32), phases, cap
    raise ValueError(f"insufficient safety pairs: {len(edges)} < {limit}")


def _fast_reference(query_index: int, round_index: int, train: list[dict],
                    calibration: Mapping[str, Any], fps: Mapping[str, Any],
                    buckets: Mapping[str, Mapping[str, list[int]]],
                    parent_orders: Mapping[str, list[str]], usage: Counter,
                    seen: Mapping[int, set[str]], cap: int = 96) -> tuple[int, bool] | None:
    query = train[query_index]; bucket = query["pair_bucket_key"]
    parents = parent_orders[bucket]; start = int(stable_sha256(
        f"v28:window:{query['record_id']}:{round_index}")[:12], 16) % len(parents)
    window = [parents[(start + offset) % len(parents)]
              for offset in range(min(128, len(parents)))]
    desired = bool((int(stable_sha256(query["record_id"])[-2:], 16) + round_index) % 2)
    choices = []
    for parent in window:
        if parent == query["parent_id"] or parent in seen[query_index]:
            continue
        candidates = [index for index in buckets[bucket][parent] if usage[index] < cap]
        if not candidates:
            continue
        reference = min(candidates, key=lambda index: (usage[index], stable_sha256(
            f"v28:record:{query['record_id']}:{train[index]['record_id']}")))
        near = _near(train[reference], query, calibration[bucket])
        similarity = DataStructs.TanimotoSimilarity(fps[query["parent_id"]], fps[parent])
        choices.append((near == desired, similarity, -usage[reference],
                        stable_sha256(train[reference]["record_id"]), reference, near))
    if not choices:
        return None
    selected = max(choices)
    return selected[-2], selected[-1]


def _near(reference: Mapping[str, Any], query: Mapping[str, Any],
          calibration: Mapping[str, Any]) -> bool:
    if query["measurement_kind"] == "binary":
        return query["category_id"] == reference["category_id"]
    sd = float(calibration.get("logit_calibration_sample_standard_deviation")
               or calibration["calibration_sample_standard_deviation"])
    return abs(float(reference["_effective_geometry_value"])
               - float(query["_effective_geometry_value"])) / sd <= calibration["numeric_center_sd"]


def _fingerprints(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for row in rows:
        if row["parent_id"] in output:
            continue
        molecule = Chem.MolFromSmiles(row["parent_smiles"])
        if molecule is None:
            raise ValueError(f"invalid frozen parent SMILES: {row['parent_smiles']}")
        output[row["parent_id"]] = MORGAN.GetFingerprint(molecule)
    return output

