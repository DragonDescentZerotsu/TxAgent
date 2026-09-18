"""Read the prepared retrieval plan used by progressive prompts.

This is the first of the three progressive runtime stages.  Offline publishers
do the expensive chemistry, scaffold, mapping, and score work.  At prediction
time ``load_candidates`` reads only the cache ``VERSION.json`` and indexed
SQLite rows, returning the L1 molecule cards, later-level record deltas, and an
audit receipt.  Morgan and assay-transfer are ranking policies over the same
immutable cache; this module never calls a model and never renders a prompt.
"""

from __future__ import annotations

from typing import Any

from predict.retrieval.assay_reranking.cache_matched import (
    DEFAULT_CACHE_BUNDLE,
    DEFAULT_GOLD_CONTEXT_MAPPING,
    CONTEXT_L2_CACHE_BUNDLE,
    CONTEXT_L2_WEIGHTED_CACHE_BUNDLE,
    CONTEXT_L2_MORGAN_SEMANTIC_CACHE_BUNDLE,
    CONTEXT_L2_MORGAN_SEMANTIC_V2_CACHE_BUNDLE,
    CONTEXT_L2_MORGAN_SEMANTIC_V3_CACHE_BUNDLE,
    CONTEXT_L2_MORGAN_SEMANTIC_V4_CACHE_BUNDLE,
    INDIRECT_MORGAN_SEMANTIC_CACHE_BUNDLE,
    INDIRECT_MORGAN_SEMANTIC_V2_CACHE_BUNDLE,
    INDIRECT_MORGAN_SEMANTIC_V3_CACHE_BUNDLE,
    L1_CONTEXT_CACHE_BUNDLE,
    SEMANTIC_BUCKET_CACHE_BUNDLE,
    load_cache_policy,
    load_candidates as _load_candidates,
)

def load_candidates(*args: Any, **kwargs: Any):
    """Load one cache-backed prompt plan without rebuilding retrieval assets."""
    return _load_candidates(*args, **kwargs)


__all__ = [
    "DEFAULT_CACHE_BUNDLE",
    "DEFAULT_GOLD_CONTEXT_MAPPING",
    "CONTEXT_L2_CACHE_BUNDLE",
    "CONTEXT_L2_WEIGHTED_CACHE_BUNDLE",
    "CONTEXT_L2_MORGAN_SEMANTIC_CACHE_BUNDLE",
    "CONTEXT_L2_MORGAN_SEMANTIC_V2_CACHE_BUNDLE",
    "CONTEXT_L2_MORGAN_SEMANTIC_V3_CACHE_BUNDLE",
    "CONTEXT_L2_MORGAN_SEMANTIC_V4_CACHE_BUNDLE",
    "INDIRECT_MORGAN_SEMANTIC_CACHE_BUNDLE",
    "INDIRECT_MORGAN_SEMANTIC_V2_CACHE_BUNDLE",
    "INDIRECT_MORGAN_SEMANTIC_V3_CACHE_BUNDLE",
    "L1_CONTEXT_CACHE_BUNDLE",
    "SEMANTIC_BUCKET_CACHE_BUNDLE",
    "load_cache_policy",
    "load_candidates",
]
