from __future__ import annotations
import numpy as np
from assay_transfer.targets import stable_sha256
from collections import Counter, defaultdict
from typing import Any, Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
import math, statistics, json
def trim_binary(edges, rows):
    groups = defaultdict(list)
    usage = Counter(int(r) for _, r, _ in edges)
    for i, (q, _, _) in enumerate(edges):
        if rows[q]["measurement_kind"] == "binary":
            groups[int(q)].append(i)
    keep = np.ones(len(edges), dtype=bool)
    for q in sorted(groups, key=lambda k: rows[k]["record_id"]):
        indices = groups[q]
        near = sum(int(edges[i, 2]) for i in indices)
        far = len(indices) - near
        majority = int(near > far)
        remove = max(near, far) - 3 * min(near, far)
        if remove <= 0:
            continue
        candidates = [i for i in indices if edges[i, 2] == majority]
        candidates.sort(key=lambda i: (-usage[int(edges[i, 1])], stable_sha256(
            f"v24.1:trim:{rows[q]['record_id']}:{rows[edges[i, 1]]['record_id']}")))
        for i in candidates[:remove]:
            keep[i] = False
            usage[int(edges[i, 1])] -= 1
    return edges[keep]
