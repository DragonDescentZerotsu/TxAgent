"""Explicit TDC profile; conditioned inputs and scientific contracts stay unchanged."""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

from .json_utils import sha256_file

PROFILE = 'tdc_scaffold_train_labels.v1'


@lru_cache(maxsize=8)
def load_manifest(path, digest):
    path = Path(path)
    if sha256_file(path) != digest:
        raise ValueError('TDC benchmark manifest changed')
    value = json.loads(path.read_text())
    if value['profile'] != PROFILE:
        raise ValueError('unsupported TDC benchmark profile')
    for task in value['tasks'].values():
        for item in task.values():
            if isinstance(item, dict) and 'path' in item and 'sha256' in item:
                if sha256_file(Path(item['path'])) != item['sha256']:
                    raise ValueError(f'TDC input changed: {item["path"]}')
    return value


def task_contract(contract, profile=''):
    if not profile:
        return contract
    if profile != PROFILE:
        raise ValueError(f'unknown benchmark profile: {profile}')
    if contract.task == 'ames':
        return replace(contract, endpoint_name='molecule-level Ames bacterial reverse mutagenicity',
            label_scope='tdc_ames_molecule.v1', system_role='Assess bacterial reverse-mutation evidence.',
            task_instructions=(
                'Predict whether the query molecule is Ames mutagenic. No query strain panel or S9 activation condition is supplied; do not invent one.',
                'Preserve strain and activation restrictions of individual source results. A negative result for one strain or limited panel does not establish a global negative.',
                'Other genetic-damage endpoints and genotoxicity mechanisms may inform the prediction when transferable; explain the mechanistic link and limitations. They are not measured Ames outcomes.',
                'Dataset-label records report a molecular classification, not an independently specified assay or evidence of positivity in every strain.',
                *(s.replace('requested bacterial condition', 'query molecule') for s in contract.task_instructions[4:8]),
            ), evidence_grounding_rules=tuple((k, v.replace('requested bacterial condition', 'query molecule'))
                for k, v in contract.evidence_grounding_rules))
    if contract.task == 'carcinogens':
        instructions = tuple(s for s in contract.task_instructions
            if not s.startswith(('Predict source-supported', 'The benchmark pools', 'When the query')))
        return replace(contract, endpoint_name='rodent carcinogenicity', label_scope='tdc_rodent_carcinogenicity.v1',
            task_instructions=('Predict the query molecule\'s rodent carcinogenicity. No specific exposure condition is supplied; do not invent dose, route, duration or rodent strain.', *instructions))
    return contract
