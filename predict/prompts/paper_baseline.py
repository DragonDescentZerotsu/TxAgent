"""Frozen paper-baseline rendering and response validation; no retrieval or model calls."""

from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import yaml

from predict.prompts import bundle_directory, template_environment
from predict.utils.json import parse_json_content
from predict.utils.response import response_validation_errors, structured_response_is_valid

BUNDLE = "paper_baseline_v1"
RECORD_RENDERING = "exact_duplicate_fields.v1"
TDC_PROFILE = "tdc_scaffold_train_labels.v1"
ENVIRONMENT = template_environment()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def profiles():
    return yaml.safe_load((bundle_directory(BUNDLE) / "tasks.yaml").read_text())


def profile(payload):
    definition = payload["task_definition"]
    task, benchmark = definition["task"], definition.get("benchmark_profile", "")
    if benchmark not in ("", TDC_PROFILE):
        raise ValueError("unknown paper baseline benchmark profile")
    spec = profiles()["conditioned"][task]
    if benchmark:
        spec = profiles()["tdc_overrides"].get(task, spec)
    if payload["required_json_schema"] != spec["required_json_schema"]:
        raise ValueError("paper baseline output schema changed")
    if payload["protocol"]["citation_rule"] != spec["citation_rule"]:
        raise ValueError("paper baseline citation instructions changed")
    return spec


def _deduplicated_details(evidence, card):
    """Remove only exact copies of named fields already visible on this card.

    Unknown fields, empty values, differing normalized/raw values and all
    multi-example records are retained. No truncation or semantic rewriting.
    This operates after identity sanitization and never changes the payload.
    """
    details = deepcopy(evidence)

    def displayed(field):
        return bool(card[field]) and (field != 'reported_unit' or card['reported_value'] is not None)

    def same_value(left, right):
        return type(left) is type(right) and json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)

    def remove_copy(path, field, *, singleton=False):
        value = card[field]
        # Falsy values may be hidden by the template; leave them in the JSON.
        if not displayed(field):
            return
        expected = [value] if singleton else value
        parent = details
        for part in path[:-1]:
            if not isinstance(parent, dict) or part not in parent:
                return
            parent = parent[part]
        if (isinstance(parent, dict) and path[-1] in parent
                and same_value(parent[path[-1]], expected)):
            del parent[path[-1]]

    for path, field in (
        (('endpoint', 'name'), 'endpoint'),
        (('endpoint', 'measurement', 'value'), 'reported_value'),
        (('endpoint', 'measurement', 'unit'), 'reported_unit'),
        (('group', 'endpoint_group'), 'evidence_family'),
        (('text', 'evidence'), 'support_text'),
    ):
        remove_copy(path, field)
    for source, field in (('assay_context', 'assay_context'),
                          ('species_context', 'species'),
                          ('qualifying_conditions', 'qualifying_conditions')):
        remove_copy(('annotations', 'scope', source), field, singleton=True)
    examples = details.get('examples', [])
    if len(examples) == 1 and isinstance(examples[0], dict):
        # Temporarily address the one source record without flattening its fields.
        source = examples[0]
        for raw, field in (('endpoint_type', 'endpoint'), ('reported_value', 'reported_value'),
                           ('reported_units', 'reported_unit'), ('assay_context', 'assay_context'),
                           ('species_context', 'species'), ('qualifying_conditions', 'qualifying_conditions'),
                           ('support_text', 'support_text')):
            value = card[field]
            if displayed(field) and raw in source and same_value(source[raw], value):
                del source[raw]
    return details


def render(payload):
    rendering = payload.get('protocol', {}).get('record_rendering')
    if rendering not in (None, RECORD_RENDERING):
        raise ValueError(f'unsupported record rendering: {rendering}')
    env = ENVIRONMENT
    task = profile(payload)['task']
    system = env.get_template(f'{BUNDLE}/system.jinja').render(**task, mode='morgan',
        query_prior_visible=bool(payload.get('query_prior')),
        output_instructions='Return exactly one valid JSON object matching the requested schema. '
            'Supportive records support your selected final prediction; contradictory records support the alternative. '
            + payload['protocol']['citation_rule'])
    molecules = []
    for number, neighbor in enumerate(payload['neighbors'], 1):
        cards = []
        for row in neighbor['records']:
            evidence = row['evidence']
            raw = (evidence.get('examples') or [{}])[0]
            measurement = evidence['endpoint']['measurement']
            card = dict(card_id=row['card_id'], level=row.get('level', ''),
                evidence_family=evidence['group']['endpoint_group'],
                endpoint=evidence['endpoint']['name'], reported_value=measurement['value'], reported_unit=measurement['unit'],
                assay_context=raw.get('assay_context', ''), species=raw.get('species_context', ''),
                qualifying_conditions=raw.get('qualifying_conditions', ''),
                extra_details=evidence, support_text=evidence['text']['evidence'])
            if rendering == RECORD_RENDERING:
                card['extra_details'] = _deduplicated_details(evidence, card)
            cards.append(card)
        molecules.append(dict(number=number, canonical_smiles=neighbor.get('canonical_smiles', 'not shown'),
                              morgan_similarity=neighbor['similarity'], evidence_cards=cards))
    user = env.get_template(f'{BUNDLE}/user.jinja').render(task_id=payload['task_definition']['task'],
        query={'canonical_smiles': 'not shown', **payload['query']},
        query_prior=payload.get('query_prior', {}), sections=[{'heading': '', 'molecules': molecules}],
        required_json_schema=payload['required_json_schema'])
    return [{'role': 'system', 'content': system.strip()}, {'role': 'user', 'content': user.strip()}]


def citation_errors(content, *, task, visible_card_ids):
    errors: list[str] = []
    partitions: dict[str, list[str]] = {}
    for field in ("supportive_card_ids", "contradictory_card_ids"):
        raw = content.get(field)
        if not isinstance(raw, list):
            errors.append(f"{field} must be an array")
            partitions[field] = []
        else:
            partitions[field] = [str(value) for value in raw]
            if len(partitions[field]) != len(set(partitions[field])):
                errors.append(f"{field} contains duplicates")
    union = set().union(*(set(values) for values in partitions.values()))
    total = sum(len(values) for values in partitions.values())
    if not union <= visible_card_ids:
        errors.append("supportive/contradictory cards must be visible")
    if total != len(union):
        errors.append("supportive/contradictory card lists must be disjoint")

    basis_raw = content.get("prediction_basis_card_ids")
    basis = {str(value) for value in basis_raw} if isinstance(basis_raw, list) else set()
    if not isinstance(basis_raw, list):
        errors.append("prediction_basis_card_ids must be an array")
    if not basis <= visible_card_ids:
        errors.append("prediction_basis_card_ids contains unknown card IDs")
    if not basis <= union:
        errors.append("prediction basis cards must be supportive or contradictory")

    prediction = str(content.get(task['prediction_field']) or "")
    if prediction not in {task['positive_prediction'], task['negative_prediction']}:
        errors.append(f"invalid {task['prediction_field']}")
    claims = content.get("claims")
    if not isinstance(claims, list):
        errors.append("claims must be an array")
    else:
        if len(claims) > 8:
            errors.append("claims must contain at most 8 items")
        for claim in claims:
            refs = claim.get("card_ids") if isinstance(claim, Mapping) else None
            if (
                not isinstance(refs, list)
                or not set(map(str, refs)) <= visible_card_ids
            ):
                errors.append("every claim must reference only visible card IDs")
                break

    gaps = content.get("evidence_gaps")
    if isinstance(gaps, list) and len(gaps) > 6:
        errors.append("evidence_gaps must contain at most 6 items")
    if not isinstance(content.get("decision_summary"), str) or not content["decision_summary"].strip():
        errors.append("decision_summary must be nonempty text")
    if not isinstance(gaps, list) or any(not isinstance(s, str) for s in gaps):
        errors.append("evidence_gaps must be a text array")
    return errors


def validation_kwargs(payload, aliases):
    task = profile(payload)["task"]
    return {
        "required_fields": tuple(payload["required_json_schema"]),
        "allowed_values": {task["prediction_field"]: {task["positive_prediction"], task["negative_prediction"]},
                           "confidence": {"high", "moderate", "low"}},
        "content_validator": lambda content: citation_errors(content, task=task, visible_card_ids=set(aliases)),
    }


def verify_request(prepared, manifest):
    """Validate the frozen task, request and selected-record bindings."""
    if prepared.get("version") != "direct_indirect_budgeted.v3" or prepared.get("prompt_version") != "evidence_v5":
        raise ValueError("unsupported paper baseline prepared format")
    payload = prepared["prompt_payload"]
    spec = profile(payload)
    task = payload["task_definition"]["task"]
    if manifest["task_contracts"][task] != spec["contract_sha256"]:
        raise ValueError("paper baseline task contract changed")
    if manifest.get("benchmark_profile", "") != payload["task_definition"].get("benchmark_profile", ""):
        raise ValueError("paper baseline benchmark binding changed")
    if prepared["manifest_hash"] != content_hash(manifest):
        raise ValueError("paper baseline manifest changed")
    if prepared["request_hash"] != content_hash(prepared["messages"]) or render(payload) != prepared["messages"]:
        raise ValueError("paper baseline request or render changed")
    aliases = prepared["card_alias_map"]
    cards = [row["card_id"] for neighbor in payload["neighbors"] for row in neighbor["records"]]
    if len(cards) != len(aliases) or set(cards) != set(aliases):
        raise ValueError("paper baseline card aliases changed")
    ids = prepared["direct_ids"] + prepared["indirect_ids"]
    if (len(ids) != len(aliases) or set(ids) != set(aliases.values())
            or len(prepared["direct_ids"]) != prepared["n_direct"]
            or len(prepared["indirect_ids"]) != prepared["n_indirect"]):
        raise ValueError("paper baseline record membership changed")


def verify_prepared(prepared, manifest, output):
    """Check a frozen request/response and return its original binary decision."""
    verify_request(prepared, manifest)
    payload, aliases = prepared['prompt_payload'], prepared['card_alias_map']
    spec = profile(payload)
    if output.get("status") != "ok" or output.get("request_hash") != prepared["request_hash"]:
        raise ValueError("paper baseline response binding changed")
    response = output.get("llm", {})
    if "raw_content" in response and parse_json_content(response["raw_content"]) != response.get("content"):
        raise ValueError("paper baseline raw response parses differently")
    errors = response_validation_errors(response, **validation_kwargs(payload, aliases))
    if not structured_response_is_valid(response) or errors:
        raise ValueError(f"invalid paper baseline response: {errors}")
    value = str(response["content"][spec["task"]["prediction_field"]])
    return int(value == spec["task"]["positive_prediction"])


def verify_batch(root):
    """Read-only verification of saved requests, responses, labels and metrics."""
    from sklearn.metrics import accuracy_score, f1_score

    root = Path(root)
    manifest_path = root / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["inputs"]:
        raise ValueError("paper baseline manifest has no tasks")
    reports = []
    for task in manifest["inputs"]:
        summary = root / "summary" / task
        saved = [json.loads(line) for line in (summary / "predictions.jsonl").read_text().splitlines() if line.strip()]
        if not saved or len({r["index"] for r in saved}) != len(saved):
            raise ValueError("empty or repeated paper baseline query indices")
        labels, predictions = [], []
        for row in saved:
            query = root / "runs" / task / f"query_{row['index']:05d}"
            prepared = json.loads((query / "prepared.json").read_text())
            if prepared["prompt_payload"]["task_definition"]["task"] != task:
                raise ValueError("paper baseline query task changed")
            output = json.loads((query / "output.json").read_text())
            prediction = verify_prepared(prepared, manifest, output)
            label = json.loads((query / "evaluation.json").read_text())["label"]
            if prediction != row["pred_label"] or label != row["label"]:
                raise ValueError("paper baseline saved labels differ")
            labels.append(label)
            predictions.append(prediction)
        values = {"macro_f1": f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0),
                  "accuracy": accuracy_score(labels, predictions)}
        metrics = json.loads((summary / "metrics.json").read_text())
        if len(saved) != metrics["n_total"] or metrics["n_successful"] != len(saved):
            raise ValueError("paper baseline summary coverage differs")
        if any(abs(values[k] - metrics[k]) > 1e-12 for k in values):
            raise ValueError("paper baseline saved metrics differ")
        reports.append({"task": task, "queries": len(saved), **values})
    return {"status": "verified", "scope": "frozen requests and saved responses; no new retrieval or inference",
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "tasks": reports}


def project_records(payload, records):
    """Rebuild source cards while preserving a frozen query, prior and task contract."""
    from predict.utils.evidence import minimal_evidence_from_row

    profile(payload)
    if not payload['query'].get('canonical_smiles'):
        raise ValueError('paper baseline preparation requires visible query structures')
    neighbors, alias_map = {}, {}
    ordered = sorted(records, key=lambda r: (r['kind'] != 'direct', -r['similarity'], r.get('knn_rank', 0), r['id']))
    for i, record in enumerate(ordered, 1):
        molecule = record['native_molecule_id']
        alias = f'C{i:03d}'
        alias_map[alias] = record['id']
        neighbor = neighbors.setdefault(molecule, {
            'molecule': f'neighbor_{len(neighbors)+1}', 'similarity': round(record['similarity'], 2),
            'records': []})
        neighbor['canonical_smiles'] = record['canonical_smiles']
        raw = {k: v for k, v in record['card'].items() if k not in ('evidence_family', 'evidence_family_level')}
        if record.get('record_smiles'):
            raw['source_molecule_smiles'] = record['record_smiles']
        evidence = minimal_evidence_from_row({
            'evidence_source': record.get('evidence_source', 'Starling'), 'molecule_id': molecule,
            'canonical_smiles': record.get('record_smiles', record['canonical_smiles']), 'molecule_names': record.get('molecule_names', []),
            'endpoint_group': record['group'], 'standard_type': raw['endpoint_type'],
            'standard_value': raw['reported_value'], 'standard_units': raw['reported_units'],
            'assay_description': raw['support_text'], 'source_record_examples': [raw],
            'evidence_scope': {'assay_context': [raw['assay_context']], 'species_context': [raw['species_context']],
                               'qualifying_conditions': [raw['qualifying_conditions']]}})
        evidence['molecule'] = {'id': neighbor['molecule']}
        neighbor['records'].append({'card_id': alias, 'evidence_kind': record['kind'], 'evidence': evidence})
    levels = {r['id']: r['card'].get('evidence_family_level', '') for r in ordered}
    for neighbor in neighbors.values():
        for row in neighbor['records']:
            row['level'] = levels[alias_map[row['card_id']]]
    payload = deepcopy(payload)
    payload['neighbors'] = list(neighbors.values())
    return payload, alias_map
