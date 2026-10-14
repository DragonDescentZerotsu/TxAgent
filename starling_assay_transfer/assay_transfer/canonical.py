"""Load the canonical six-task policy independently of historical builders."""

import argparse
import hashlib
import json
import math
from pathlib import Path

DEFAULT_CONTRACT = Path(__file__).with_suffix('.json')
TASKS = ('BBB_Martins', 'Bioavailability_Ma', 'Skin_Reaction', 'Ames', 'Carcinogens', 'DILI')
TASK_FIELDS = {
    'slug', 'initialization_source', 'record_levels', 'evaluation_levels',
    'record_degree_cap', 'record_degree_caps_by_level', 'context_direct_degree_cap',
    'context_indirect_degree_cap', 'variance_ratio_maximum', 'ood_bucket_records',
    'holdout_policy', 'calibration_policy', 'variance_gate_fit',
    'record_sampling_policy', 'evaluation_policy',
}
COMMON_FIELDS = {
    'evidence_library_version', 'gold_version', 'source_resolution', 'record_pair_scope',
    'context_direct_pair_scope', 'training_same_parent_pairs', 'record_degree_cap_scope',
    'context_direct_degree_scope', 'variance_ratio_statistic', 'variance_ratio_undefined',
    'numeric_target', 'record_binary_target', 'context_calibration_policy',
    'indirect_context_splits', 'query_measurement_visible', 'test_values_fit_targets',
    'completion_requires',
}


def _validate_task(task, settings):
    if set(settings) != TASK_FIELDS:
        raise ValueError(f'{task}: missing or unknown settings: {set(settings) ^ TASK_FIELDS}')
    levels, evaluation = settings['record_levels'], settings['evaluation_levels']
    if levels is not None and (not isinstance(levels, list) or not levels or
                              any(not isinstance(level, str) for level in levels) or len(set(levels)) != len(levels)):
        raise ValueError(f'{task}: record_levels must be a unique list or null for all pinned levels')
    if not isinstance(evaluation, list) or any(not isinstance(level, str) for level in evaluation):
        raise ValueError(f'{task}: evaluation_levels must be a list of strings')
    if len(set(evaluation)) != len(evaluation) or (levels is not None and not set(evaluation) <= set(levels)):
        raise ValueError(f'{task}: evaluation levels must be unique and included in record levels')
    overrides = settings['record_degree_caps_by_level']
    if not isinstance(overrides, dict) or (levels is not None and not set(overrides) <= set(levels)):
        raise ValueError(f'{task}: degree overrides must refer to included record levels')
    caps = [settings[key] for key in ('record_degree_cap', 'context_direct_degree_cap', 'context_indirect_degree_cap')]
    if any(type(cap) is not int or cap < 2 or cap % 2 for cap in caps + list(overrides.values())):
        raise ValueError(f'{task}: degree caps must be positive even integers')
    if settings['context_direct_degree_cap'] % 4:
        raise ValueError(f'{task}: direct context degree must be divisible by four')
    ratio = settings['variance_ratio_maximum']
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio < 0:
        raise ValueError(f'{task}: variance maximum must be finite and nonnegative')
    bounds = settings['ood_bucket_records']
    if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 2 or
                              any(type(bound) is not int for bound in bounds) or not 1 <= bounds[0] <= bounds[1]):
        raise ValueError(f'{task}: OOD sizes must be an inclusive [minimum, maximum] or null')
    if bool(evaluation) != (bounds is not None):
        raise ValueError(f'{task}: OOD settings and record evaluation levels must be enabled together')
    for key in ('slug', 'initialization_source', 'holdout_policy', 'calibration_policy',
                'variance_gate_fit', 'record_sampling_policy', 'evaluation_policy'):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f'{task}: {key} must be explicit')


def load_contract(path=DEFAULT_CONTRACT):
    document = json.loads(Path(path).read_text())
    if set(document) != {'schema_version', 'common', 'tasks'} or document['schema_version'] != 'assay_transfer.canonical_contract.v1':
        raise ValueError('Unsupported canonical contract schema')
    common = document['common']
    if set(common) != COMMON_FIELDS:
        raise ValueError(f'Missing or unknown common settings: {set(common) ^ COMMON_FIELDS}')
    if common['evidence_library_version'] != 'v10' or common['gold_version'] != 'v1':
        raise ValueError('The canonical contract requires V10 evidence and Gold-v1')
    if common['source_resolution'] != 'immutable_manifest_and_file_hashes':
        raise ValueError('Canonical inputs require immutable manifests and file hashes')
    if common['variance_ratio_undefined'] != 'reject' or common['test_values_fit_targets'] is not False:
        raise ValueError('Undefined variance ratios are rejected; test values cannot fit targets')
    if common['training_same_parent_pairs'] is not False or common['query_measurement_visible'] is not False:
        raise ValueError('Training cannot pair the same parent or expose query measurements')
    if common['indirect_context_splits'] != ['train']:
        raise ValueError('Indirect evidence belongs only in context training')
    target = common['numeric_target']
    if set(target) != {'center_sd', 'temperature_sd', 'clipped'} or target['clipped'] is not False:
        raise ValueError('Numeric targets require explicit center, temperature and unclipped probabilities')
    for key in ('center_sd', 'temperature_sd'):
        value = target[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    required = {'input_hashes', 'effective_contract_hash', 'split_audit', 'artifact_hashes', 'coverage', 'verification_receipt'}
    if set(common['completion_requires']) != required or len(common['completion_requires']) != len(required):
        raise ValueError('Completion requires all provenance, split, coverage and verification checks')
    if set(document['tasks']) != set(TASKS):
        raise ValueError('The canonical contract must explicitly include all six tasks')
    for task, settings in document['tasks'].items():
        _validate_task(task, settings)
    if len({settings['slug'] for settings in document['tasks'].values()}) != len(TASKS):
        raise ValueError('Task slugs must be unique')
    return document


def task_contract(task, path=DEFAULT_CONTRACT, *, split_disjointness='scaffold',
                  augmentation_ratio=None):
    document = load_contract(path)
    matches = [name for name, settings in document['tasks'].items() if task in (name, settings['slug'])]
    if len(matches) != 1:
        raise ValueError(f'Unknown or ambiguous task: {task}')
    name = matches[0]
    effective = {'schema_version': document['schema_version'], 'task': name,
                 'common': document['common'], **document['tasks'][name]}
    if split_disjointness not in ('parent', 'scaffold'):
        raise ValueError('Split disjointness must be parent or scaffold')
    effective.update(schema_version='assay_transfer.effective_contract.v2',
                     split_disjointness=split_disjointness, disjointness_scope='task')
    if augmentation_ratio is not None:
        if type(augmentation_ratio) not in (int, float) or not math.isfinite(augmentation_ratio) or augmentation_ratio < 0:
            raise ValueError('Augmentation ratio must be finite and nonnegative')
        effective['augmentation_ratio'] = float(augmentation_ratio)
    encoded = json.dumps(effective, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**effective, 'contract_sha256': hashlib.sha256(encoded).hexdigest()}


def degree_cap(task, level=None, component='record', path=DEFAULT_CONTRACT):
    settings = task_contract(task, path)
    if component not in ('record', 'context_direct', 'context_indirect'):
        raise ValueError(f'Unknown degree component: {component}')
    levels = settings['record_levels']
    if level is not None and levels is not None and level not in levels:
        raise ValueError(f'Unsupported record level: {task}/{level}')
    cap = settings[f'{component}_degree_cap']
    return settings['record_degree_caps_by_level'].get(level, cap) if component == 'record' else cap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument('--task', help='Canonical task name or record-level slug')
    args = parser.parse_args()
    document = task_contract(args.task, args.contract) if args.task else load_contract(args.contract)
    print(json.dumps(document, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
