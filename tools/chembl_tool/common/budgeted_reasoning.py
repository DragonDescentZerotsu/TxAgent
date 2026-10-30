"""One decision over a fixed direct set and selected indirect records."""
import json

from .evidence_contract import minimal_evidence_from_row
from .identity_blind import _hide_structures, _is_identity_term, _replace_identity_terms
from .reasoning_validation import evidence_validation_kwargs as validation_kwargs
from .reasoning_payload import external_condition_sentence

VERSION = 'budgeted_reasoning.v1'


def build_messages(contract, query, records, query_tools, analog_tools, *, visibility_mode,
                   prompt_version='budgeted_v1', query_prior=None, return_payload=False,
                   record_rendering='exact_duplicate_fields.v1'):
    """Only source evidence and tool text are visible; selection scores are not."""
    blind = visibility_mode == 'identity_blind'
    aligned = prompt_version == 'full_flat_v5'
    if prompt_version not in ('budgeted_v1', 'full_flat_v5'):
        raise ValueError('unsupported budgeted prompt')
    if visibility_mode not in ('identity_blind', 'deployment_visible_prefetched'):
        raise ValueError('unsupported visibility mode')
    neighbors, alias_map = {}, {}
    ordered = sorted(records, key=lambda r: (r['kind'] != 'direct', -r['similarity'], r.get('knn_rank', 0), r['id']))
    def tool_text(result):
        return {'tool': result.get('tool_name'), 'text': str(result.get('content') or '')}
    for i, record in enumerate(ordered, 1):
        molecule = record['native_molecule_id']
        alias = f'C{i:03d}'
        alias_map[alias] = record['id']
        neighbor = neighbors.setdefault(molecule, {
            'molecule': f'neighbor_{len(neighbors)+1}', 'similarity': round(record['similarity'], 2),
            'tools': [tool_text(t) for t in analog_tools.get(molecule, [])], 'records': []})
        if not blind:
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
    if aligned:
        for neighbor in neighbors.values():
            neighbor.pop('tools')
        levels = {r['id']: r['card'].get('evidence_family_level', '') for r in ordered}
        for neighbor in neighbors.values():
            for row in neighbor['records']:
                row['level'] = levels[alias_map[row['card_id']]]
    schema = {contract.prediction_field: ' | '.join(sorted(contract.prediction_values)),
              'confidence': 'high | moderate | low', 'supportive_card_ids': [], 'contradictory_card_ids': [],
              'prediction_basis_card_ids': [], 'claims': [{'claim': 'source-grounded statement or explicit inference', 'card_ids': []}],
              'evidence_gaps': [], 'decision_summary': 'Concise decision and treatment of relevant counterevidence.'}
    payload = {'protocol': {'version': VERSION, 'mode': 'single decision',
        'evidence_rule': 'Attribute observations to the tested neighbor and condition. Explain transfer to the query. '
                         'Repeated descriptions are not independent experiments. Distinguish predictions, mechanisms and measured outcomes. '
                         'Assess positive and negative findings by the same standards. Missing evidence is not a negative outcome.',
        'citation_rule': 'Cite only supplied card IDs. Basis cards must also be supportive or contradictory. '
                         'Structure/property inferences may have empty card_ids and must be identified as inferences. '
                         'Maximum 8 claims and 6 evidence gaps; empty lists are valid.',
        **dict(contract.evidence_grounding_rules)},
        'task_definition': {'task': contract.task, 'endpoint': contract.endpoint_name, 'label_scope': contract.label_scope,
                            'instructions': list(contract.task_instructions)},
        'query': {'molecule': 'query', 'tools': [tool_text(query_tools)]},
        'neighbors': list(neighbors.values()), 'required_json_schema': schema}
    if query.get('benchmark_profile'):
        payload['task_definition']['benchmark_profile'] = query['benchmark_profile']
    condition = external_condition_sentence(query)
    if condition:
        payload['query']['external_condition'] = condition
    if not blind:
        payload['query']['canonical_smiles'] = query['canonical_smiles']
    if aligned:
        from .flat_prompt_alignment import RECORD_RENDERING, project_prior
        payload['query'].pop('tools')
        payload['query_prior'] = project_prior(query_prior)
        if record_rendering not in ('full_fields', RECORD_RENDERING):
            raise ValueError('unsupported record rendering')
        if record_rendering != 'full_fields':
            payload['protocol']['record_rendering'] = record_rendering
    if blind:
        terms = [query.get(k, '') for k in ('drug', 'molecule_name', 'molecule_id', 'standard_inchi_key')]
        structures = [query['canonical_smiles']]
        for r in records:
            terms.extend([r['native_molecule_id'], *r.get('molecule_names', [])])
            structures.append(r['canonical_smiles'])
        terms = sorted({str(t) for t in terms if _is_identity_term(str(t))}, key=len, reverse=True)
        payload = _replace_identity_terms(_hide_structures(payload, tuple(structures)), terms)
    messages = [{'role': 'system', 'content': contract.system_role + ' Make one prediction from the supplied evidence and computed properties. '
             'Do not identify the query by name. Return one JSON object matching the schema; do not reveal hidden chain-of-thought.'},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    if aligned:
        from .flat_prompt_alignment import render
        messages = render(contract, payload)
    return (messages, alias_map, payload) if return_payload else (messages, alias_map)


def prepared_payload(prepared):
    """Read the audit payload only after checking it renders the actual messages."""
    if prepared.get('prompt_version', 'budgeted_v1') == 'budgeted_v1':
        return json.loads(prepared['messages'][1]['content'])
    from importlib import import_module
    from .flat_prompt_alignment import render
    payload = prepared['prompt_payload']
    task = payload['task_definition']['task']
    contract = import_module(f'tools.chembl_tool.tasks.{task}.experiment_config').get_progressive_task_contract()
    from .tdc_benchmark import task_contract
    contract = task_contract(contract, payload['task_definition'].get('benchmark_profile', ''))
    if render(contract, payload) != prepared['messages']:
        raise ValueError('prompt audit payload differs from the actual rendered messages')
    return payload
