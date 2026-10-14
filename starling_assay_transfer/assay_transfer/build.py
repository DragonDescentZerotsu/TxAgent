"""Shared input pinning and immutable-stage helpers for the separate builders."""

import argparse
from pathlib import Path

from .storage import (PACKAGE, digest, environment, implementation, read_json, read_parquet,
                        stage, verify_inputs, verify_manifest, write_json, write_parquet)
from .canonical import DEFAULT_CONTRACT, TASKS, task_contract
from .inputs import FILES, pin

SUPPORTED = read_json(PACKAGE / 'assets/scientific_policies.json')


def supported(policy):
    expected = SUPPORTED[policy['task']]
    tunable = {'record_degree_cap','record_degree_caps_by_level','context_direct_degree_cap',
               'context_indirect_degree_cap','variance_ratio_maximum','ood_bucket_records','contract_sha256','schema_version'}
    for key, value in expected.items():
        if key not in tunable and key != 'common' and policy[key] != value:
            raise ValueError(f'Unsupported task policy: {key}={policy[key]}')
    common = dict(policy['common'])
    common['numeric_target'] = expected['common']['numeric_target']
    if common != expected['common']:
        raise ValueError('Unsupported common scientific policy')
    if policy['context_direct_degree_cap'] % 4:
        raise ValueError('Direct context degree must be divisible by four')


def arguments(family):
    parser = argparse.ArgumentParser(description=f'Build canonical {family} datasets independently')
    parser.add_argument('command', choices=('pool','sample','augment','targets','dataset','verify','upload','all'))
    parser.add_argument('--contract', type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument('--task', action='append', help='Canonical task or slug; repeat to select tasks')
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--split-disjointness', choices=('parent','scaffold'), default='scaffold')
    parser.add_argument('--render-workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Representative scope; never a full release')
    if family == 'context':
        parser.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument('--augmentation-ratio', type=float, default=1.0, help='Indirect training examples per direct example')
    parser.add_argument('--repo-id',help='HF dataset repository; required only by upload')
    parser.add_argument('--revision',help='HF branch; defaults to a branch named by the verified build hash')
    parser.add_argument('--private',action='store_true',help='Create a private HF repository')
    args = parser.parse_args()
    if args.command=='upload' and (not args.repo_id or not args.task or len(args.task)!=1):
        parser.error('upload requires --repo-id and exactly one --task')
    if args.render_workers < 1:
        parser.error('--render-workers must be positive')
    if family == 'context':
        task_contract('DILI', args.contract, augmentation_ratio=args.augmentation_ratio)
    return args


def selected_inputs(args, gold_only=False):
    inputs = read_json(args.input_manifest)
    expected = {key:value for key,value in inputs.items() if key != 'inputs_sha256'}
    if digest(expected) != inputs['inputs_sha256']:
        raise ValueError('Pinned input manifest signature mismatch')
    tasks = [task_contract(task, args.contract)['task'] for task in args.task or TASKS]
    selected = {'schema_version':inputs['schema_version'], 'tasks':{}}
    for task in tasks:
        files = inputs['tasks'][task]['files']
        if gold_only:
            files = {role:item for role,item in files.items() if role.startswith('gold_')}
        required = {'gold_train','gold_valid','gold_test'}
        if not gold_only:
            required |= set(FILES) | {'geometry_policy'}
        missing = required - files.keys()
        if missing:
            raise ValueError(f'{task}: pinned inputs missing required roles: {sorted(missing)}')
        selected['tasks'][task] = {'files':files}
    selected['inputs_sha256'] = digest(selected)
    verify_inputs(selected)
    return selected


def dependencies(inputs, policy, smoke, family):
    return {'inputs_sha256':inputs['inputs_sha256'], 'contract':policy['contract_sha256'],
            'implementation_sha256':implementation(), 'environment':environment(),
            'scope':'smoke' if smoke else 'full', 'builder':family}


def checked_stage(path,expected,task_root):
    manifest=verify_manifest(path)
    for key,value in expected.items():
        if manifest['build_dependencies'].get(key)!=value:
            raise ValueError(f'Incompatible upstream stage: {path}: {key}')
    for name,signature in manifest['build_dependencies'].get('upstream',{}).items():
        if digest(checked_stage(task_root/name,expected,task_root))!=signature:
            raise ValueError(f'Upstream manifest changed: {name}')
    return manifest


def run_stage(root,name,base,upstream,produce,resume):
    dependencies={**base,'upstream':{parent:digest(checked_stage(root/parent,base,root)) for parent in upstream}}
    return stage(root/name,dependencies,produce,resume)


def write_receipt(root, receipt):
    write_json(root / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description='Pin evidence and Gold inputs for the separate builders')
    parser.add_argument('command', choices=('pin',))
    parser.add_argument('--contract', type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument('--task', action='append')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--gold-only', action='store_true', help='Pin Gold splits and sidecars without reading evidence libraries')
    args = parser.parse_args()
    tasks = [task_contract(task, args.contract)['task'] for task in args.task or TASKS]
    pin(args.source_root, args.input_manifest, args.contract, tasks, args.gold_only)


if __name__ == '__main__':
    main()
