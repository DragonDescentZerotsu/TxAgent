from __future__ import annotations
from assay_transfer.targets import prepare_geometry, stable_sha256
from collections import Counter, defaultdict
from typing import Any, Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
import math, statistics, json, heapq
PARENT_CAP = 4096

def allocate(parents, limits, used, other, order, coverage=False):
    """Water-fill each parent's remaining budget, prioritizing total-degree zero."""
    quotas = dict(used)
    for ids in parents.values():
        remaining = PARENT_CAP - sum(used[key] for key in ids)
        heap = [(int(used[key] + other[key] > 0), used[key], order[key], key)
                for key in ids if used[key] < limits[key]]
        heapq.heapify(heap)
        while heap and remaining > 0:
            _, count, tie, key = heapq.heappop(heap)
            count += 1
            quotas[key] = count
            remaining -= 1
            if not coverage and count < limits[key]:
                heapq.heappush(heap, (1, count, tie, key))
    return Counter(quotas)

def initialize(rows, calibration, caps, usage):
    records = prepare_geometry({r["record_id"]: r for r in rows
                                if r["dataset_split"] == "train"}, calibration)
    parents, buckets = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for key, row in records.items():
        parents[row["parent_id"]].append(key)
        buckets[row["pair_bucket_key"]][row["parent_id"]].append(key)
    order = {key: stable_sha256(f"v24:fair:{key}") for key in records}
    limits = {role: {} for role in usage}
    for key, row in records.items():
        group = buckets[row["pair_bucket_key"]]
        degree = caps[row["pair_bucket_key"]]["resolved_degree_cap"]
        limits["query"][key] = min(degree, len(group) - 1)
        limits["reference"][key] = min(degree, sum(map(len, group.values())) - len(group[row["parent_id"]]))
    return {"records": records, "parents": parents, "buckets": buckets, "order": order,
            "limits": limits, "usage": usage, "seen": defaultdict(set), "near": Counter()}

def reference_priority(key, state, targets):
    usage = state["usage"]
    return (int(usage["query"][key] + usage["reference"][key] > 0),
            usage["reference"][key] / max(1, targets[key]), usage["reference"][key],
            state["order"][key], key)

def available_reference(heap, state, quotas, targets):
    # Keys only increase within a pass, so stale minima can be refreshed lazily.
    while heap:
        key = heap[0][-1]
        if state["usage"]["reference"][key] >= quotas[key]:
            heapq.heappop(heap)
            continue
        priority = reference_priority(key, state, targets)
        if priority != heap[0]:
            heapq.heapreplace(heap, priority)
            continue
        return state["records"][key], priority
    return None
