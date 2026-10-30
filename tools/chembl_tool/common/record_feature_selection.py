"""Opt-in, label-free record coverage using apricot FeatureBasedSelection.

Features describe supplied card text, not inferred scientific claims. The same
encoder handles structured and free-text cards across tasks and family levels.
It is intentionally a lexical baseline, not an endpoint/direction classifier.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping, Sequence

POLICY = "feature_based.v1"
APRICOT_VERSION = "0.6.1"
FIELDS = ("endpoint", "assay_context", "species", "qualifying_conditions", "reported_value")
# Administrative fields are not scientific coverage. Never encode card IDs,
# benchmark labels, inferred evidence_direction, or the source's group IDs.
_SKIP_KEYS = {"needs_more_context", "confidence", "confidence_score", "source_unit_uncertainty",
              "liver_injury_identifier", "cancer_identifier"}
_MISSING = re.compile(r"(?:unspecified|unknown|none|null|missing|not reported|not applicable)(?:\s*\(\d+\))?", re.I)
_TOKEN = re.compile(r"[a-zα-ω][a-zα-ω0-9]*(?:[-/][a-zα-ω0-9]+)*|[<>≤≥]?[+-]?\d+(?:\.\d+)?")
_STOP = {"the", "a", "an", "of", "and", "or", "in", "to", "for", "with", "by", "was", "were", "is", "are"}


def _segments(value: Any):
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    for part in text.split("|"):
        match = re.match(r"\s*([a-z][a-z0-9_]*)\s*=\s*(.*)", part)
        key, content = (match.group(1), match.group(2)) if match else ("text", part)
        if key in _SKIP_KEYS:
            continue
        # Remove complete missing-value placeholders only; preserve 'none of
        # the animals developed tumors' and other scientific negations.
        content = "; ".join(s.strip() for s in content.split(";")
                            if s.strip() and not _MISSING.fullmatch(s.strip()))
        if content:
            yield key, content


@lru_cache(maxsize=32768)
def _encode(values: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
    features = {}
    for field, value in zip(FIELDS, values):
        block = set()
        for key, content in _segments(value):
            tokens = _TOKEN.findall(content.replace("_", " "))
            prefix = f"{field}/{key}:"
            block.update(prefix + token for token in tokens if token not in _STOP)
            # Adjacent pairs preserve local qualifiers such as 'not increased'.
            # They do NOT solve negation, subject attribution, or synonymy.
            block.update(prefix + a + " " + b for a, b in zip(tokens, tokens[1:])
                         if a not in _STOP or b not in _STOP)
        if block:
            # Under sqrt, each populated field contributes exactly one unit
            # for a single isolated card, regardless of field verbosity.
            value_per_feature = 1.0 / len(block) ** 2
            features.update((feature, value_per_feature) for feature in block)
    return tuple(sorted(features.items()))


def record_features(card: Mapping[str, Any]) -> dict[str, float]:
    return dict(_encode(tuple(str(card.get(field) or "") for field in FIELDS)))


@lru_cache(maxsize=256)
def _candidate_matrix(values: tuple[tuple[str, ...], ...]):
    """Reuse identical molecule/level candidate matrices across queries."""
    import numpy as np
    from scipy.sparse import csr_matrix

    rows = [dict(_encode(row)) for row in values]
    names = sorted({name for row in rows for name in row})
    columns = {name: i for i, name in enumerate(names)}
    rr, cc, vv = [], [], []
    for i, row in enumerate(rows):
        for name, value in row.items():
            rr.append(i)
            cc.append(columns[name])
            vv.append(value)
    X = csr_matrix((vv, (rr, cc)), shape=(len(rows), len(names)), dtype=np.float64)
    return X, names, columns


def feature_matrix(cards: Sequence[Mapping[str, Any]],
                   covered_cards: Sequence[Mapping[str, Any]] = ()):
    """Return candidate CSR X, dense initial coverage row, and column names.

    Vocabulary is fitted on eligible candidates only. Covered cards contribute
    only to these same columns, so no historical/future-only feature can enter
    the candidate set. Returned X/names are cached and must be treated read-only.
    No labels or training data are needed.
    """
    import numpy as np

    X, names, columns = _candidate_matrix(tuple(
        tuple(str(card.get(field) or "") for field in FIELDS) for card in cards))
    covered = np.zeros((1, len(names)), dtype=np.float64)
    for card in covered_cards:
        for name, value in record_features(card).items():
            if name in columns:
                covered[0, columns[name]] += value
    return X, covered, names


@lru_cache(maxsize=1)
def selection_contract() -> dict[str, Any]:
    """Fail explicitly if the optional dependency is absent or unpinned."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        installed = version("apricot-select")
    except PackageNotFoundError as exc:
        raise RuntimeError("feature_based.v1 requires optional apricot-select==0.6.1; "
                           "see RECORD_FEATURE_SELECTION.md") from exc
    if installed != APRICOT_VERSION:
        raise RuntimeError(f"feature_based.v1 requires apricot-select=={APRICOT_VERSION}, got {installed}")
    return {"policy": POLICY, "apricot_version": installed, "concave_func": "sqrt",
            "optimizer": "naive", "features": "field_scoped_unigrams_bigrams.v1",
            "fields": list(FIELDS), "field_normalization": "inverse_squared_feature_count",
            "history_scope": "all_previously_selected_cards_in_query",
            "molecule_order": "existing_new_pool_then_active_pool_order",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


@lru_cache(maxsize=1)
def _selector_type():
    """Reuse apricot's stateless JIT kernel across small, independent fits.

    apricot 0.6.1 recompiles its gain and unused streaming kernels on every fit.
    This CSR-only adapter retains its BaseSelection initialization, objective,
    and greedy optimizer. Tests compare both ranking and gains with its public
    FeatureBasedSelection API. No global library functions are patched.
    """
    import numpy as np
    from apricot import FeatureBasedSelection
    from apricot.functions.base import BaseSelection
    from apricot.functions.featureBased import calculate_gains_sparse, sparse_dtypes

    kernel = calculate_gains_sparse(np.sqrt, sparse_dtypes, False, True, False)

    class CachedFeatureBasedSelection(FeatureBasedSelection):
        def _initialize(self, X):
            BaseSelection._initialize(self, X)
            self.current_values = self.initial_subset.sum(axis=0).astype("float64")
            self.current_concave_values = np.sqrt(self.current_values)
            self.current_concave_values_sum = self.current_concave_values.sum()
            self.calculate_gains_ = kernel

    return CachedFeatureBasedSelection


def select_feature_cards(ordered_cards: Sequence[Mapping[str, Any]], limit: int, *,
                         covered_cards: Sequence[Mapping[str, Any]] = ()):
    """Select up to limit eligible cards; caller supplies deterministic tie order.

    Returned cards remain raw. The audit is selection-only and must never be
    rendered as evidence. Prior cards are represented by a 2D initial_subset,
    not mixed into X where they could become selectable again.
    """
    import numpy as np
    contract = selection_contract()
    cards = list(ordered_cards)
    if limit < 0:
        raise ValueError("card limit must be nonnegative")
    X, initial, names = feature_matrix(cards, covered_cards)
    k = min(limit, len(cards))
    if not k or not names:
        ranking = list(range(k))
        gains = [0.0] * k
    else:
        selector = _selector_type()(k, concave_func="sqrt", optimizer="naive",
                                    initial_subset=initial, random_state=0)
        selector.fit(X)
        ranking = selector.ranking.tolist()
        gains = selector.gains.tolist()
        assert len(ranking) == k and len(set(ranking)) == k
        assert np.isfinite(gains).all()
    audit = {"policy": contract["policy"], "n_candidates": len(cards), "n_features": len(names),
             "n_covered_cards": len(covered_cards), "n_featureless_candidates": int(np.count_nonzero(X.getnnz(axis=1) == 0)),
             "vocabulary_sha256": hashlib.sha256(json.dumps(names).encode()).hexdigest(),
             "ranking_card_ids": [str(cards[i]["card_id"]) for i in ranking], "gains": gains}
    return [dict(cards[i]) for i in ranking], audit
