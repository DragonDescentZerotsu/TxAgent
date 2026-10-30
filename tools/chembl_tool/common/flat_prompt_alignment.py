"""Pinned upstream full-flat prose, task mapping and portable prompt export.

Task science remains in experiment_config; this module only adapts presentation.
It does not retrieve evidence, call models, or change benchmark labels.
"""
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import yaml

from .json_utils import sha256_file, write_json_atomic

VERSION = 'full_flat_v5_adapted.v2'
RECORD_RENDERING = 'exact_duplicate_fields.v1'
UPSTREAM = '3c36c2640ada12119f4a74a9bdf8511b0d7b8f6e'
ASSETS = Path(__file__).with_name('prompts') / 'full_flat_v5'
TASKS = ('bbb_martins', 'bioavailability_ma', 'skin_reaction', 'ames', 'dili', 'carcinogens')
PRIOR_FIELDS = (
    'passive_bbb_plausibility', 'efflux_or_transporter_prior',
    'oral_bioavailability_prior', 'absorption_prior', 'solubility_or_dissolution_prior',
    'metabolism_or_clearance_prior', 'endpoint_prior', 'structural_alerts',
    'skin_sensitization_prior', 'reactive_or_haptenation_prior', 'activation_prior', 'skin_exposure_context',
    'ames_prior', 'activation_dependence',
    'confidence', 'reasoning_summary',
    'property_drivers', 'caveats',
)


@lru_cache(maxsize=1)
def assets():
    return yaml.safe_load((ASSETS / 'tasks.yaml').read_text())


def task_prompt(contract, *, portable=False):
    """Use upstream BBB/Bio science and map the other current task contracts."""
    if contract.task in assets():
        result = deepcopy(assets()[contract.task])
        if not portable and contract.task == 'bioavailability_ma':
            result['task_instructions'] = [s.replace('Predict pass when', 'Predict high when')
                .replace('and fail when', 'and low when').replace('Predict fail only', 'Predict low only')
                for s in result['task_instructions']]
    else:
        instructions = [s for s in contract.task_instructions if not s.startswith('Choose ')]
        instructions.extend(text for _, text in contract.evidence_grounding_rules)
        instructions = [s.replace('progressive update', 'prediction')
                         .replace('overturn a prior Ames prediction', 'inform an Ames prediction')
                         .replace('for a justified update', 'for a justified prediction')
                         .replace('for an update', 'for a prediction') for s in instructions]
        result = dict(prompt_profile=contract.label_scope, system_role=contract.system_role,
                      endpoint_name=contract.endpoint_name, label_scope=contract.label_scope,
                      task_instructions=instructions)
        if portable:
            result['task_instructions'].insert(0, f'Predict pass for {contract.positive_prediction} '
                f'and fail for {contract.negative_prediction} under the task definition.')
    if portable:
        result.update(prediction_field='final_prediction', positive_prediction='pass', negative_prediction='fail',
            required_fields=['claims', 'final_prediction'], required_json_schema={'claims': [{
                'claim': 'concise evidence-grounded statement', 'molecule_ids': ['Molecule N'],
                'record_ids': ['Record N-N'], 'evidence_role': 'supportive | contradictory'}]})
    else:
        result.update(prediction_field=contract.prediction_field,
                      positive_prediction=contract.positive_prediction, negative_prediction=contract.negative_prediction)
    return result


def project_prior(prior):
    """Expose single-molecule analysis only, never final labels or tool payloads."""
    if not isinstance(prior, dict) or not isinstance(prior.get('reasoning_summary'), str) or not prior['reasoning_summary'].strip():
        raise ValueError('cached prior requires a nonempty single-molecule reasoning_summary')
    return {key: prior[key] for key in PRIOR_FIELDS if prior.get(key) not in (None, '', [], {})}


@lru_cache(maxsize=16)
def _prior_index(task, root, manifest_hash):
    from .json_utils import read_jsonl
    root = Path(root)
    if sha256_file(root / 'experiment_manifest.json') != manifest_hash:
        raise ValueError('cached prior source manifest changed')
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    source = manifest['inputs'][task]
    path = Path(source['input_jsonl'])
    if sha256_file(path) != source['input_sha256']:
        raise ValueError('cached prior source input hash mismatch')
    index = {}
    for i, row in enumerate(read_jsonl(path)):
        key = (row['drug'], row.get('condition_group', ''))
        if key in index:
            raise ValueError('duplicate cached prior query identity')
        index[key] = i
    return index


def load_prior(task, query, source):
    """Reuse a frozen single analysis by exact structure AND condition."""
    if source.get('format') == 'single_analysis_cache.v1':
        cache = _single_cache(source['path'], source['sha256'])
        if cache['task'] != task:
            raise ValueError('single analysis task mismatch')
        key = (query['drug'], query.get('condition_group', ''))
        return project_prior(cache['by_query'][key]), dict(source)
    root = Path(source['root'])
    index = _prior_index(task, str(root), source['manifest_sha256'])
    key = (query['drug'], query.get('condition_group', ''))
    if key not in index:
        raise ValueError(f'no cached single-molecule analysis for {task} query/condition')
    path = root / task / 'queries' / f'query_idx{index[key]:05d}' / 'levels/level_1/prepared.json'
    if sha256_file(path) != source['prepared_sha256'].get(str(index[key])):
        raise ValueError('cached prior snapshot changed after manifest binding')
    value = json.loads(path.read_text())
    if (value.get('query_smiles'), value.get('condition_group', '')) != key:
        raise ValueError(f'cached prior structure/condition mismatch: {path}')
    if not value.get('tool_prefetch_complete'):
        raise ValueError(f'incomplete source preparation: {path}')
    prior = project_prior(value.get('query_prior'))
    return prior, {'path': str(path), 'sha256': sha256_file(path),
                   'root': str(root), 'manifest_sha256': source['manifest_sha256']}


def prior_sources(path, tasks):
    if not path:
        raise ValueError('upstream alignment requires --cached-prior-sources mapping tasks to frozen analysis files or progressive roots')
    mapping = json.loads(Path(path).read_text())
    result = {}
    for task in tasks:
        root = Path(mapping[task]).resolve()
        if root.is_file():
            digest = sha256_file(root)
            cache = _single_cache(str(root), digest)
            if cache['task'] != task:
                raise ValueError('single analysis task mismatch')
            result[task] = {'format': 'single_analysis_cache.v1', 'path': str(root), 'sha256': digest,
                            'benchmark_profile': cache['benchmark_profile'], 'task_contract_hash': cache['task_contract_hash']}
            continue
        manifest = root / 'experiment_manifest.json'
        snapshots = {str(int(p.parents[2].name.removeprefix('query_idx'))): sha256_file(p)
                     for p in (root / task / 'queries').glob('query_idx*/levels/level_1/prepared.json')}
        if not snapshots:
            raise ValueError(f'no frozen query-prior snapshots for {task}: {root}')
        result[task] = {'root': str(root), 'manifest_sha256': sha256_file(manifest), 'prepared_sha256': snapshots}
    return result


@lru_cache(maxsize=16)
def _single_cache(path, digest):
    from .budgeted_evidence import content_hash
    from .json_utils import read_jsonl
    path = Path(path)
    if sha256_file(path) != digest:
        raise ValueError('single analysis cache changed')
    value = json.loads(path.read_text())
    if value['version'] != 'single_analysis_cache.v1':
        raise ValueError('unsupported single analysis cache')
    source = value['input']
    if sha256_file(Path(source['path'])) != source['sha256']:
        raise ValueError('single analysis input changed')
    expected = {(r['drug'], r.get('condition_group', '')) for r in read_jsonl(Path(source['path']))}
    entries = {}
    for row in value['rows']:
        key = (row['query_smiles'], row['condition_group'])
        if key in entries:
            raise ValueError('duplicate single analysis query')
        if row['query_hash'] != content_hash(list(key)):
            raise ValueError('single analysis query binding mismatch')
        entries[key] = project_prior(row['analysis'])
    if set(entries) != expected:
        raise ValueError('single analysis coverage mismatch')
    return {**value, 'by_query': entries}


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


def render(contract, payload):
    rendering = payload.get('protocol', {}).get('record_rendering')
    if rendering not in (None, RECORD_RENDERING):
        raise ValueError(f'unsupported record rendering: {rendering}')
    env = Environment(loader=FileSystemLoader(ASSETS), undefined=StrictUndefined, autoescape=False)
    task = task_prompt(contract)
    system = env.get_template('system.jinja').render(**task, mode='morgan',
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
    user = env.get_template('user.jinja').render(task_id=contract.task,
        query={'canonical_smiles': 'not shown', **payload['query']},
        query_prior=payload.get('query_prior', {}), sections=[{'heading': '', 'molecules': molecules}],
        required_json_schema=payload['required_json_schema'])
    return [{'role': 'system', 'content': system.strip()}, {'role': 'user', 'content': user.strip()}]


def export_bundle(destination, contracts):
    """Export upstream-compatible assets and an explicit bidirectional label map."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tasks = {c.task: task_prompt(c, portable=True) for c in contracts}
    (destination / 'tasks.yaml').write_text(yaml.safe_dump(tasks, sort_keys=False, allow_unicode=True))
    output = ('Return exactly one valid JSON object matching the requested schema. Return one or more concise claims '
              'using as many claims as needed to represent the experimental evidence that materially informs the decision. '
              'Each claim may cite any number of molecules, records, or both. At least one identifier is required per claim. '
              'Return only `claims` and `final_prediction`. `supportive` means evidence supporting your selected final '
              'prediction; `contradictory` means evidence supporting the alternative.')
    (destination / 'system.jinja').write_text((ASSETS / 'system.jinja').read_text().replace('{{ output_instructions }}', output))
    (destination / 'user.jinja').write_text((ASSETS / 'user.jinja').read_text())
    write_json_atomic(destination / 'provenance.json', dict(version=VERSION, upstream_commit=UPSTREAM,
        status='offline_validated_not_evaluated', supports_tasks=list(tasks),
        task_contracts={c.task: asdict(c) for c in contracts},
        label_mapping={c.task: {'local_field': c.prediction_field,
            'pass': c.positive_prediction, 'fail': c.negative_prediction} for c in contracts},
        files={name: sha256_file(destination / name) for name in ('system.jinja', 'user.jinja', 'tasks.yaml')}))
    return destination


if __name__ == '__main__':
    import argparse
    from importlib import import_module
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', required=True)
    args = parser.parse_args()
    contracts = [import_module(f'tools.chembl_tool.tasks.{task}.experiment_config').get_progressive_task_contract()
                 for task in TASKS]
    print(export_bundle(args.export_dir, contracts))
