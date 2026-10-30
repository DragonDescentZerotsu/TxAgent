"""Fixed-budget evidence selection, independent of paths, providers and labels."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import heapq
import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation

from .record_budget import (
    VERSION, DEFAULT_BUDGET, SIMPLE_POLICIES, POLICIES, SelectionConfig,
    content_hash, select_simple_records as _select_simple,
)
from .record_diversity import normalize_endpoint
from .record_selection_context import endpoint_relevance, record_quality

FEATURE_VERSION = "budgeted_record_features.v3"
BUDGET_GRID = (16, 64, 128)
WEIGHT_GRID = ((1, 0, 0), (0, .75, .25), (.25, .5625, .1875),
               (.5, .375, .125), (.75, .1875, .0625), (.5, .5, 0), (.5, .25, .25))




def normalized_features(task, prepared, annotation=None):
    """Field-independent content features with fixed per-block feature mass.

    Native aliases and repeated canonical/raw fields must not create diversity.
    Semantic tags use the same token space, not a separate schema-specific one.
    This is a transparent lexical/observation representation, not a claim that
    arbitrary scientific paraphrases are semantically equivalent.
    """
    from .record_feature_selection import _segments, _TOKEN, _STOP
    from .record_diversity import CALL_FIELDS, assessment_source_input
    source = deepcopy(prepared)
    fields = source["input"]["source_fields"]
    # Processing flags and duplicate numerical representations are not evidence
    # diversity. Prefer the source's already standardized value/unit pair.
    outcome_fields = source.get('groups', {}).get('outcome', {})
    for key in ('embedded_unit', 'finite_scalar_value'):
        outcome_fields.pop(key, None)
    if fields.get('canonical_measurement_text') and fields.get('canonical_unit_text'):
        outcome_fields.pop('measurement_text', None)
        outcome_fields.pop('unit_text', None)
        if '±' in fields['canonical_measurement_text']:
            outcome_fields.pop('variation_value', None)
    for key, value in list(source.get("groups", {}).get("endpoint", {}).items()):
        if endpoint_relevance(task, value, fields) == "unrelated":
            del source["groups"]["endpoint"][key]
    if annotation:
        annotation = deepcopy(annotation)
        # These free-text notes can include citations, interpretations and
        # background arms; they are not typed exposure features. Bound model
        # observations/tags retain the reported conditions. The actual source
        # paragraph gets its own block so repeated reader-visible text cannot
        # have zero overlap merely because native endpoint fields differ.
        for key in ('qualifying_conditions', 'extra_details'):
            source.get('groups', {}).get('exposure', {}).pop(key, None)
        source['groups']['source_content'] = {'support_text':source['input'].get('support_text', '')}
        annotation["outcomes"] = [o for o in annotation.get("outcomes", [])
            if o.get("focal_scope") == "focal"
            and endpoint_relevance(task, o["endpoint"], fields) != "unrelated"]
        # An assessed readout supersedes lossy native endpoint/direction labels.
        # Numerical measurements remain source values, never model estimates.
        source['groups']['endpoint'] = {str(i):o['endpoint'] for i,o in enumerate(annotation['outcomes'])}
        visible_fields = (assessment_source_input(source['input'])['source_fields']
                          if annotation_is_assessed(annotation) else fields)
        source['groups']['outcome'] = {
            key:value for key,value in outcome_fields.items()
            if key in {'canonical_measurement_text','canonical_unit_text','measurement_text','unit_text'}
            and key in visible_fields
            and fields.get('canonical_unit_text') != 'binary_outcome_class'
        }
        for group in ("endpoint", "outcome"):
            annotation[group] = [tag for tag in annotation.get(group, [])
                                if endpoint_relevance(task, tag, fields) != "unrelated"]
    def terms(text):
        # Endpoint spelling normalization must not erase a minus sign in a
        # number or exponent. Normalize numbers before the lexical tokenizer.
        text = unicodedata.normalize('NFKC', str(text)).lower().replace('_',' ').replace('−','-')
        text = re.sub(r'(?<=[a-z])-(?=[a-z])', ' ', text)
        def number(match):
            try:
                return format(Decimal(match[0]).normalize(), 'g')
            except InvalidOperation:
                return match[0]
        text = re.sub(r'(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?(?![\w.])', number, text)
        for pattern, replacement in (
            (r'\brattus norvegicus\b|\brats\b', 'rat'),
            (r'\bmus musculus\b|\bmice\b', 'mouse'),
            (r'\bhomo sapiens\b|\bhumans\b', 'human'),
            (r'\bp glycoprotein\b|\bp gp\b|\bpgp\b|\babcb1\b', 'p glycoprotein'),
            (r'\barea under (?:the )?(?:concentration time )?curve\b', 'auc'),
        ):
            text = re.sub(pattern, replacement, text)
        tokens = _TOKEN.findall(text)
        return {t for t in tokens if t not in _STOP} | {
            a+' '+b for a,b in zip(tokens,tokens[1:]) if a not in _STOP or b not in _STOP}

    blocks = defaultdict(set)
    for group, fields in source['groups'].items():
        for value in fields.values():
            for field, content in _segments(value):
                # Some native schemas embed direction codes inside the
                # measurement string. The assessed observation owns direction;
                # keep the accompanying measurement, not a second native call.
                if annotation and group == 'outcome' and field in CALL_FIELDS:
                    continue
                blocks[group].update(terms(normalize_endpoint(content) if group == 'endpoint' else content))
        for value in (annotation or {}).get(group, []):
            blocks[group].update(terms(value))
    for outcome in (annotation or {}).get('outcomes', []):
        endpoint = normalize_endpoint(outcome['endpoint'])
        blocks['observation'].add(f'{endpoint}/{outcome["kind"]}')
        for name in ('evidence_basis', 'result_scope', 'statistical_support'):
            if outcome.get(name):
                blocks['observation'].add(f'{endpoint}/{name}={outcome[name]}')
        for name in ('conditions','comparator'):
            for span in outcome.get(name, []):
                blocks['observation'].update(f'{endpoint}/{name}/{t}' for t in terms(span['quote']))
    features = {f'{group}/{term}': 1/len(terms_)**2
                for group, terms_ in blocks.items() for term in terms_}
    scale = sum(math.sqrt(v) for v in features.values())
    return {key: value / scale**2 for key, value in features.items()} if scale else {}


def score_candidate(task, condition, candidate, prepared, annotation=None, *, transfer=1.0, uniform_quality=False):
    """Keep audit/feature fields internal; the renderer receives raw cards only."""
    if not math.isfinite(transfer) or not 0 <= transfer <= 1:
        raise ValueError("transfer score must be finite and in [0, 1]")
    quality = record_quality(task, condition, prepared, candidate["card"], annotation)
    q = 1.0 if uniform_quality else quality["score"]
    return {**candidate, "quality": q, "transfer": transfer, "utility": q * transfer,
            "features": normalized_features(task, prepared, annotation), "quality_audit": quality}


def score_bound_candidate(task, condition, candidate, binding, annotation=None, **options):
    """Ambiguous duplicate surfaces get conservative quality/common features."""
    alternatives = binding.get("alternatives", [binding["prepared"]])
    ambiguous = binding.get("ambiguous_native_metadata", False)
    scored = [score_candidate(task, condition, candidate, p,
              (annotation or {}).get('alternatives', {}).get(content_hash(p['input'])) if ambiguous else annotation, **options)
              for p in (alternatives if ambiguous else [binding["prepared"]])]
    result = min(scored, key=lambda r: r["quality"])
    if ambiguous:
        common = set.intersection(*(set(r["features"]) for r in scored))
        result["features"] = {j: min(r["features"][j] for r in scored) for j in common}
        result["quality_audit"] = {**result["quality_audit"],
            "ambiguous_native_metadata": True, "policy": "minimum_quality_common_features",
            "alternative_scores": [r["quality"] for r in scored]}
    return {**result, "source_record_ids": binding["source_record_ids"],
            "source_input_hashes": binding["source_input_hashes"]}


def annotation_is_assessed(annotation):
    if not annotation:
        return False
    if 'alternatives' in annotation:
        return bool(annotation['alternatives']) and all(annotation_is_assessed(a) for a in annotation['alternatives'].values())
    return bool(annotation.get('outcomes')) and all('evidence_basis' in o for o in annotation['outcomes'])


class _Objective:
    def __init__(self, weights):
        self.weights = weights
        self.features = defaultdict(float)
        self.molecule_group = defaultdict(float)
        self.groups = defaultdict(float)
        self.components = [0.0, 0.0, 0.0]

    def gains(self, record):
        molecule, group, p = record["molecule_id"], record["group"], record["utility"]
        diversity = sum(math.sqrt(self.features.get((molecule, j), 0) + p*p*x)
                        - math.sqrt(self.features.get((molecule, j), 0)) for j, x in record["features"].items())
        increment = max(0.0, p*p - self.molecule_group.get((molecule, group), 0))
        coverage = math.sqrt(self.groups.get(group, 0) + increment) - math.sqrt(self.groups.get(group, 0))
        return p, diversity, coverage

    def add(self, record, gains):
        molecule, group, p = record["molecule_id"], record["group"], record["utility"]
        for j, x in record["features"].items():
            self.features[molecule, j] += p*p*x
        old = self.molecule_group[molecule, group]
        self.molecule_group[molecule, group] = max(old, p*p)
        self.groups[group] += max(0.0, p*p-old)
        self.components = [v + d for v, d in zip(self.components, gains)]

    def total(self, gains):
        return sum(w * g for w, g in zip(self.weights, gains))


def objective_value(records, weights=(.5, .375, .125)):
    objective = _Objective(weights)
    for record in records:
        objective.add(record, objective.gains(record))
    return objective.total(objective.components)




def select_records(records, config=SelectionConfig()):
    """Deterministic lazy greedy for Joint, with simple matched pool controls.

    The heap contains admissible marginal upper bounds because all scores and
    features are fixed and the objective is submodular. Tie order never uses a
    label or outcome direction. Random policies use stable per-record hashes.
    """
    records = list(records)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate candidate IDs")
    for r in records:
        if not r["molecule_id"] or not r["group"]:
            raise ValueError("each record requires one molecule and one group")
    if config.policy in SIMPLE_POLICIES:
        return _select_simple(records, config)
    for r in records:
        if not math.isfinite(r["utility"]) or not 0 <= r["utility"] <= 1:
            raise ValueError("record utility must be finite and in [0,1]")
        if any(not math.isfinite(x) or x < 0 for x in r["features"].values()):
            raise ValueError("features must be finite and nonnegative")
        if sum(math.sqrt(x) for x in r["features"].values()) > 1 + 1e-9:
            raise ValueError("features must have whole-record square-root mass <= 1")
    order = sorted(range(len(records)), key=lambda i: (-records[i].get("similarity", 0), records[i]["id"]))
    objective = _Objective(config.weights)
    selected, trace = [], []
    def add(i):
        r = records[i]
        gains = objective.gains(r)
        objective.add(r, gains)
        selected.append(r)
        trace.append({"record_id": r["id"], "quality": r["quality"], "transfer": r["transfer"],
                      "utility": r["utility"], "gain": objective.total(gains), "component_gains": list(gains)})
    limit = min(config.budget, len(records))
    if config.policy == "joint":
        heap = [(-objective.total(objective.gains(records[i])), rank, i, 0)
                for rank, i in enumerate(order)]
        heapq.heapify(heap)
        while heap and len(selected) < limit:
            negative, rank, i, stamp = heapq.heappop(heap)
            if stamp != len(selected):
                gain = objective.total(objective.gains(records[i]))
                heapq.heappush(heap, (-gain, rank, i, len(selected)))
            elif -negative > 0:
                add(i)
            else:
                break
    elif config.policy == "quality":
        for i in sorted(order, key=lambda i: -records[i]["utility"])[:limit]:
            if records[i]["utility"] > 0:
                add(i)
    return selected, {"version": VERSION, "policy": config.policy, "budget": config.budget,
        "weights": list(config.weights), "seed": config.seed, "n_candidates": len(records),
        "n_selected": len(selected), "shortfall": config.budget-len(selected),
        "n_molecules": len({r["molecule_id"] for r in selected}),
        "n_groups": len({r["group"] for r in selected}), "components": objective.components,
        "objective": objective.total(objective.components), "selection_trace": trace}
