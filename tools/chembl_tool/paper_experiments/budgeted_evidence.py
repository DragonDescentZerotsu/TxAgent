"""Budgeted mode of the family runner; shared transport, tools and scheduler."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import gc
import json
import math
from pathlib import Path

from tools.chembl_tool.common.budgeted_evidence import (
    BUDGET_GRID, DEFAULT_BUDGET, FEATURE_VERSION, POLICIES, SIMPLE_POLICIES, SelectionConfig, content_hash, score_bound_candidate, select_records, annotation_is_assessed,
)
from tools.chembl_tool.common.budgeted_retrieval import (
    REGISTRY, bind_source_records, build_evidence_view, cached_annotations, retrieve_candidates, source_paths,
    write_feature_inputs,
)
from tools.chembl_tool.common.budgeted_preparation import evidence_view
from tools.chembl_tool.common.budgeted_reasoning import build_messages, validation_kwargs
from tools.chembl_tool.common.reasoning_validation import response_validation_errors
from tools.chembl_tool.common import flat_prompt_alignment
from tools.chembl_tool.common.json_utils import read_jsonl, sha256_file, write_json_atomic
from tools.chembl_tool.common.retrieval_features import load_retrieval_index
from tools.chembl_tool.common.starling.conditioned_benchmark import task_root
from tools.chembl_tool.paper_experiments import run_conditioned_assay_progressive_curve as runtime

from tools.chembl_tool.paper_experiments.prepared_evidence import (
    VERSION, task_contract, _run_query, _run_prepared, summarize,
)


def add_arguments(parser):
    parser.add_argument('--run-prepared', action='store_true', help='Run independently manifest-bound prepared budgeted requests in output-root; do not rebuild retrieval.')
    parser.add_argument('--benchmark-manifest', default='', help='Explicit TDC input/index/train-label manifest; conditioned benchmark remains the default.')
    parser.add_argument('--direct-knn-root', default='', help='TDC-only control: replay each task Morgan KNN top-3 from this ML result root; no additional Direct filtering.')
    parser.add_argument('--budgeted-prompt', choices=('full_flat_v5', 'budgeted_v1'), default='full_flat_v5',
                        help='upstream prose + cached single analysis, or the historical tool-visible JSON prompt.')
    parser.add_argument('--cached-prior-sources', default='', help='JSON mapping task to a frozen progressive root containing query_prior snapshots.')
    parser.add_argument('--record-rendering', choices=('full_fields', flat_prompt_alignment.RECORD_RENDERING),
                        default=flat_prompt_alignment.RECORD_RENDERING,
                        help='Keep full_fields only when matching a frozen adapter-v1 comparison.')
    parser.add_argument('--evidence-setting', choices=('none', 'direct', 'direct_indirect'), default='')
    parser.add_argument('--indirect-selector', choices=POLICIES, default='joint')
    parser.add_argument('--indirect-identity-policy', choices=('parent_disjoint', 'scaffold_disjoint'),
                        default='parent_disjoint', help='Query-relative exclusion for indirect candidates, applied before neighbor ranking.')
    parser.add_argument('--indirect-budget', type=int, default=DEFAULT_BUDGET)
    parser.add_argument('--joint-weights', type=float, nargs=3, default=(.5, .375, .125), metavar=('A', 'D', 'H'))
    parser.add_argument('--selection-seed', type=int, default=0)
    parser.add_argument('--records-per-molecule', type=int, default=5, help='Molecule-cap and group-balanced controls; Joint has a global budget.')
    parser.add_argument('--candidate-molecules', type=int, default=128, help='Fixed global indirect neighbor pool, independent of record budget.')
    parser.add_argument('--min-similarity', type=float, default=.3)
    parser.add_argument('--direct-records-per-molecule', type=int, default=5)
    parser.add_argument('--direct-selector', choices=('neighbor_fill', 'molecule_cap'), default='molecule_cap')
    parser.add_argument('--direct-budget', type=int, default=50, help='Maximum total direct records, independent of the indirect budget.')
    parser.add_argument('--split-scheme', choices=('scaffold', 'random'), default='scaffold')
    parser.add_argument('--evaluation-subset', choices=('valid', 'test'), default='valid')
    parser.add_argument('--retrieval-index-root', default='')
    parser.add_argument('--visibility-mode', choices=('identity_blind', 'deployment_visible_prefetched'), default='identity_blind')
    parser.add_argument('--annotation-db', default='', help='Frozen input-bound v4 source assessment cache, required for Quality/Joint runs; simple controls use raw cards.')
    parser.add_argument('--feature-inputs-only', action='store_true', help='Export exact retrieved inputs for offline annotation; no tools or prediction.')
    parser.add_argument('--allow-native-features', action='store_true', help='Diagnostic only: allow missing assessed semantic annotations.')
    parser.add_argument('--transfer-scores', default='', help='Optional query-record JSON scores; otherwise T=1.')
    parser.add_argument('--uniform-record-quality', action='store_true', help='Ablation: set Q=1 without changing candidate eligibility.')
    parser.add_argument('--skip-tool-prefetch', action='store_true', help='Offline prepare-only diagnostic; cannot run inference.')
    parser.add_argument('--transport-max-retries', type=int, default=0)


def _read(path):
    return json.loads(Path(path).read_text())


def _pin(path, value):
    """Never overwrite a checkpoint under a different scientific contract."""
    if path.exists() and _read(path) != value:
        raise ValueError(f'changed prepared contract: {path}; use a fresh output root')
    write_json_atomic(path, value)


def _code_hashes():
    common = Path('tools/chembl_tool/common')
    files = [Path(__file__), Path(__file__).with_name('prepared_evidence.py'), *[common / name for name in (
        'tdc_benchmark.py', 'tdc_evidence.py', 'record_budget.py', 'budgeted_evidence.py', 'budgeted_retrieval.py', 'budgeted_preparation.py', 'budgeted_reasoning.py', 'record_selection_context.py',
        'record_diversity.py', 'record_numeric_features.py', 'identity_blind.py', 'evidence_contract.py', 'flat_prompt_alignment.py',
        'assay_retrieval.py', 'retrieval_policy.py', 'molecule_identity.py', 'reasoning_validation.py')],
        Path(runtime.__file__), common / 'task_workflows/retrieve_neighbors.py', common / 'retrieval_features.py',
        common / 'reasoning_payload.py', common / 'starling/external_condition.py',
        common / 'starling/new_task_identity.py', common / 'starling/new_task_retrieval_identity.py',
        *[Path('tools/chembl_tool/tasks') / name for name in (
            'bbb_martins/prepare_condition_review.py', 'bioavailability_ma/condition_ontology.py',
            'skin_reaction/context_conditioned_benchmark.py', 'carcinogens/reviewed_conditions.py')]]
    files.extend(sorted(flat_prompt_alignment.ASSETS.glob('*')))
    return {str(p): sha256_file(p) for p in files}


def task_specs(args):
    specs = runtime._progressive_task_specs(args.split_scheme, args.evaluation_subset,
        Path(args.retrieval_index_root) if getattr(args, 'retrieval_index_root', '') else None)
    if getattr(args, 'benchmark_manifest', ''):
        from tools.chembl_tool.common.tdc_benchmark import load_manifest
        data = load_manifest(args.benchmark_manifest, sha256_file(Path(args.benchmark_manifest)))
        if args.split_scheme != 'scaffold' or _uses_assessments(args) or args.feature_inputs_only:
            raise ValueError('TDC supports scaffold simple baselines only')
        specs = {t: replace(specs[t], input_jsonl=Path(v[args.evaluation_subset]['path']),
            index=Path(v['index']['path']), family_manifest=Path(v['catalog']['path']))
            for t, v in data['tasks'].items()}
    return specs




def _direct_selection(records, cap, budget=50, policy='molecule_cap', seed=0):
    # The same selector and seed apply with and without additional indirect cards.
    return select_records(records, SelectionConfig(budget=budget, policy=policy,
        per_molecule=cap, seed=seed))[0]


def _uses_assessments(args):
    return args.evidence_setting == 'direct_indirect' and args.indirect_selector not in SIMPLE_POLICIES


def _write_prepared_query(args, task, raw_query, q, manifest, pool, direct, selected, audit, candidates):
    # Gold remains outside source selection and reader messages.
    query = {'benchmark_profile': manifest.get('benchmark_profile', ''), 'canonical_smiles': raw_query['drug'], 'condition_group': raw_query.get('condition_group', '')}
    query_hash = content_hash(query)
    records = direct + selected
    analogs = {r['native_molecule_id']: {'analog_id': r['native_molecule_id'],
        'canonical_smiles': r['canonical_smiles']} for r in records}
    aligned = getattr(args, 'budgeted_prompt', 'budgeted_v1') == 'full_flat_v5'
    prior, prior_receipt = None, None
    if aligned:
        prior, prior_receipt = flat_prompt_alignment.load_prior(task, raw_query, manifest['prior_sources'][task])
    if args.skip_tool_prefetch or aligned:
        tools, failures, query_tools = {}, [], {}
    else:
        tools, failures, query_tools = runtime._prefetch_analog_tools(query_smiles=query['canonical_smiles'], analogs=analogs,
            tool_service_url=args.tool_service_url, timeout_s=args.timeout_s)
    prompt_version = getattr(args, 'budgeted_prompt', 'budgeted_v1')
    messages, aliases, payload = build_messages(task_contract(args, task), query, records, query_tools, tools,
        visibility_mode=args.visibility_mode, prompt_version=prompt_version, query_prior=prior, return_payload=True,
        record_rendering=getattr(args, 'record_rendering', flat_prompt_alignment.RECORD_RENDERING))
    prepared = {'version': VERSION, 'manifest_hash': content_hash(manifest), 'query_hash': query_hash,
        'request_hash': content_hash(messages), 'messages': messages, 'card_alias_map': aliases,
        'tool_prefetch_complete': not args.skip_tool_prefetch,
        'direct_ids': [r['id'] for r in direct], 'indirect_ids': [r['id'] for r in selected],
        'candidate_pool_hash': content_hash(sorted(r['id'] for r in pool['indirect'])),
        'n_direct': len(direct), 'n_indirect': len(selected)}
    prepared.update(prompt_version=prompt_version, context_ready=not args.skip_tool_prefetch,
                    tool_prefetch_complete=not args.skip_tool_prefetch and not aligned)
    if aligned:
        prepared.update(prompt_payload=payload, query_prior_source=prior_receipt)
    write_json_atomic(q.query_dir / 'selection.json', {'query': query, 'query_hash': query_hash, 'audit': audit,
        'direct': direct, 'selected': selected, 'candidates': candidates})
    write_json_atomic(q.query_dir / 'tools.json', {'query': query_tools, 'analogs': tools, 'failures': failures})
    write_json_atomic(q.query_dir / 'evaluation.json', {'label': int(raw_query['Y']), 'benchmark_row_id': raw_query.get('benchmark_row_id')})
    write_json_atomic(q.query_dir / 'prepared.json', prepared)
    print(f'[budgeted] {task} #{q.index}: direct={len(direct)} indirect={len(selected)}/{len(pool["indirect"])}', flush=True)


def _prepare_simple_queries(args, task, rows, queries, view, manifest, config, cache=None):
    def prepare(q):
        retrieve = cache.candidates if cache is not None else retrieve_candidates
        binding = {'task': task, 'manifest': manifest} if cache is not None else {}
        pool = retrieve(rows[q.index]['drug'], view, split_scheme=args.split_scheme,
            candidate_molecules=args.candidate_molecules, min_similarity=args.min_similarity,
            indirect_identity_policy=getattr(args, 'indirect_identity_policy', 'parent_disjoint'),
            **binding) if view is not None else {'direct': [], 'indirect': []}
        _prepare_simple_query(args, task, rows[q.index], q, view, manifest, config, pool)
    with ThreadPoolExecutor(max_workers=max(1, args.preparation_workers)) as workers:
        list(workers.map(prepare, queries))


def _prepare_simple_query(args, task, raw_query, q, view, manifest, config, pool):
    direct = _direct_selection(pool['direct'], args.direct_records_per_molecule, args.direct_budget,
                               args.direct_selector, args.selection_seed)
    matched = bool(getattr(args, 'direct_knn_root', '')) and args.evidence_setting != 'none'
    if matched:
        from tools.chembl_tool.common.tdc_evidence import matched_knn_records
        direct = matched_knn_records(task, raw_query, q.index, manifest['inputs'][task])
    candidates = pool['indirect'] if args.evidence_setting == 'direct_indirect' else []
    selected, audit = select_records(candidates, replace(config, policy=config.policy if candidates else 'neighbor_fill'))
    audit['direct_selection'] = {'policy': args.direct_selector, 'budget': args.direct_budget,
        'n_selected': len(direct), 'shortfall': args.direct_budget-len(direct),
        'per_molecule': args.direct_records_per_molecule if args.direct_selector == 'molecule_cap' else None}
    if matched:
        audit['direct_selection'] = {'policy': 'matched_morgan_knn', 'budget': 3,
            'n_selected': 3, 'shortfall': 0, 'per_molecule': 1,
            'train_indices': [r['train_index'] for r in direct]}
    audit['assessment_required'] = False
    excluded = view.get('budgeted_unresolved_parent_ids', []) if view is not None else []
    audit['unresolved_parent_exclusion'] = {'count': len(excluded), 'molecule_ids_hash': content_hash(excluded)}
    candidate_audit = [{k: v for k, v in r.items() if k not in ('card', 'molecule_names')} for r in candidates]
    _write_prepared_query(args, task, raw_query, q, manifest, pool, direct, selected, audit, candidate_audit)


def _transfer_map(path):
    if not path:
        return None
    value = _read(path)
    if value.get('version') != 'query_record_transfer.v1' or not value.get('model'):
        raise ValueError('transfer file requires version=query_record_transfer.v1 and model provenance')
    output = {}
    for row in value['scores']:
        key = (row['task'], row['query_hash'], row['record_id'])
        if key in output:
            raise ValueError('duplicate query-record transfer score')
        score = float(row['score'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('transfer scores must be finite and in [0,1]')
        output[key] = score
    return output


def _prepare_task(args, task, spec, manifest, config, transfers, cache=None):
    root = Path(args.output_root)
    rows = read_jsonl(spec.input_jsonl)
    indices = runtime._selected_indices(args, len(rows))
    queries = [runtime.PreparedQuery(task, i, root / 'runs' / task / f'query_{i:05d}') for i in indices]
    missing = [q for q in queries if not (q.query_dir / 'prepared.json').exists()]
    if not missing:
        return queries
    if not _uses_assessments(args) and not args.feature_inputs_only:
        view = None if args.evidence_setting == 'none' else (
            cache.view(task, manifest) if cache is not None else evidence_view(load_retrieval_index(spec.index), task, manifest))
        _prepare_simple_queries(args, task, rows, missing, view, manifest, config, cache)
        return queries
    pools = {}
    if args.evidence_setting != 'none':
        print(f'[budgeted] {task}: loading candidate index', flush=True)
        view = build_evidence_view(load_retrieval_index(spec.index))
        for q in missing:
            pools[q.index] = retrieve_candidates(rows[q.index]['drug'], view, split_scheme=args.split_scheme,
                candidate_molecules=args.candidate_molecules, min_similarity=args.min_similarity,
                indirect_identity_policy=getattr(args, 'indirect_identity_policy', 'parent_disjoint'))
        del view
        gc.collect()
    candidates = {r['id']: r for pool in pools.values() for kind in (
        ('direct', 'indirect') if args.evidence_setting == 'direct_indirect' else ('direct',)) for r in pool[kind]}
    bindings = bind_source_records(task, candidates.values())
    if args.feature_inputs_only:
        summary = write_feature_inputs(root / 'feature_inputs', task, bindings)
        write_json_atomic(root / 'feature_inputs' / f'{task}.summary.json', summary)
        write_json_atomic(root / 'feature_inputs' / f'{task}.scope.json', {
            'manifest_hash':content_hash(manifest),
            'queries':{str(i):{kind:[r['id'] for r in pool[kind]] for kind in ('direct','indirect')}
                       for i,pool in pools.items()},
            'candidate_sources':{key:bound['source_record_ids'] for key,bound in bindings.items()},
        })
        return []
    annotations = cached_annotations(args.annotation_db, task, bindings)
    if not args.allow_native_features:
        missing_semantics = [key for key in candidates if not annotation_is_assessed(annotations.get(key))]
        if missing_semantics:
            raise ValueError(f'{task}: {len(missing_semantics)} candidates lack input-bound v4 annotations. '
                             'Export --feature-inputs-only, annotate offline, and pass a frozen --annotation-db; '
                             '--allow-native-features is an explicitly labelled diagnostic.')
    for q in missing:
        raw_query = rows[q.index]
        # The scorer/prompt never receives the gold row or label provenance.
        query = {'benchmark_profile': manifest.get('benchmark_profile', ''), 'canonical_smiles': raw_query['drug'], 'condition_group': raw_query.get('condition_group', '')}
        query_hash = content_hash(query)
        pool = pools.get(q.index, {'direct': [], 'indirect': []})
        direct = _direct_selection(pool['direct'], args.direct_records_per_molecule, args.direct_budget,
                                   args.direct_selector, args.selection_seed)
        direct = [score_bound_candidate(task, query['condition_group'], r, bindings[r['id']], annotations.get(r['id']))
                  for r in direct]
        scored = []
        if args.evidence_setting == 'direct_indirect':
            for candidate in pool['indirect']:
                bound = bindings[candidate['id']]
                key = (task, query_hash, candidate['id'])
                if transfers is not None and key not in transfers:
                    raise ValueError(f'missing transfer score for {key}; partial transfer files cannot silently use T=1')
                transfer = 1.0 if transfers is None else transfers[key]
                record = score_bound_candidate(task, query['condition_group'], candidate, bound,
                    annotations.get(candidate['id']), transfer=transfer, uniform_quality=args.uniform_record_quality)
                scored.append(record)
        selected, audit = select_records(scored, config)
        semantic_ids = {r['id'] for r in scored if r['id'] in annotations}
        audit['semantic_features'] = {
            'annotated_candidates': len(semantic_ids),
            'native_only_candidates': len(scored) - len(semantic_ids),
            'annotated_selected': sum(r['id'] in semantic_ids for r in selected),
            'direct_records': len(direct),
            'annotated_direct': sum(annotation_is_assessed(annotations.get(r['id'])) for r in direct),
            'annotated_with_unresolved_notes': sum(bool(annotations[k].get('unresolved')) for k in semantic_ids),
        }
        records = direct + selected
        audit['semantic_features']['context_records_with_features'] = sum(bool(r['features']) for r in records)
        if not args.allow_native_features and any(not r['features'] for r in records):
            raise ValueError(f'{task} #{q.index}: a context record has no shared content features; inspect its source binding')
        candidate_audit = [{k: v for k, v in r.items() if k not in ('features', 'card', 'molecule_names')}
                           | {'features_hash': content_hash(r['features']),
                              'annotation_hash': content_hash(annotations.get(r['id']))} for r in scored]
        _write_prepared_query(args, task, raw_query, q, manifest, pool, direct, selected, audit, candidate_audit)
    return queries




def run(args, *, preparation_cache=None, preparation_submit=None):
    runtime.load_env_file(Path(args.env_file))
    if getattr(args, 'run_prepared', False):
        return _run_prepared(args)
    if args.feature_inputs_only:
        args.prepare_only = True
        args.skip_tool_prefetch = True
    if args.matched_progressive_root or args.replicate_ids or args.refresh_query_priors or args.level_reuse_source_root:
        raise ValueError('budgeted mode requires fresh independent inputs, not a legacy replay/matched suite')
    if args.skip_tool_prefetch and not args.prepare_only:
        raise ValueError('--skip-tool-prefetch is an offline prepare-only diagnostic')
    if not 1 <= args.parallelism <= args.endpoint_concurrency_budget or not 0 <= args.parallelism_per_task <= args.parallelism:
        raise ValueError('invalid shared endpoint concurrency budget')
    args.retry_race_width = args.retry_race_width or 1
    if not 1 <= args.retry_race_width <= (args.parallelism_per_task or args.parallelism):
        raise ValueError('retry race width must fit the shared pool')
    if args.max_stage_requeues < 0 or args.retry_delay_s < 0 or args.transport_max_retries < 0:
        raise ValueError('retry limits must be nonnegative')
    if args.candidate_molecules < 1 or args.direct_records_per_molecule < 1 or args.direct_budget < 0 or not 0 <= args.min_similarity <= 1:
        raise ValueError('invalid retrieval limits')
    if not _uses_assessments(args) and (args.transfer_scores or args.annotation_db or args.uniform_record_quality):
        raise ValueError('simple baselines do not use quality, transfer scores or annotation caches')
    matched = bool(getattr(args, 'direct_knn_root', ''))
    if matched and (not args.benchmark_manifest or args.direct_budget != 3 or _uses_assessments(args)):
        raise ValueError('matched KNN requires a TDC benchmark, direct budget 3, and simple selectors')
    specs = task_specs(args)
    benchmark = None
    if getattr(args, 'benchmark_manifest', ''):
        from tools.chembl_tool.common.tdc_benchmark import load_manifest
        benchmark = load_manifest(args.benchmark_manifest, sha256_file(Path(args.benchmark_manifest)))
    if not benchmark and not args.retrieval_index_root and args.evidence_setting != 'none':
        variants = _read(REGISTRY)['candidate_index_variants']
        for task, spec in list(specs.items()):
            variant = variants['source_reported_fields_v1' if task in (
                'skin_reaction', 'bioavailability_ma') else 'all_distinct_cards_v1']
            index_root = variant.get('task_roots', {}).get(task, variant['root'])
            specs[task] = replace(spec, index=Path(index_root) / task / args.split_scheme / 'assay_neighbor_index.pkl')
    args.tasks = args.tasks or list(specs)
    if set(args.tasks) - set(specs):
        raise ValueError('budgeted mode supports the six current Starling tasks')
    aligned = getattr(args, 'budgeted_prompt', 'budgeted_v1') == 'full_flat_v5'
    priors = flat_prompt_alignment.prior_sources(args.cached_prior_sources, args.tasks) if aligned and not args.feature_inputs_only else {}
    if benchmark:
        for task, source in priors.items():
            if (source.get('benchmark_profile') != benchmark['profile']
                    or source.get('task_contract_hash') != content_hash(asdict(task_contract(args, task)))):
                raise ValueError(f'{task}: TDC requires a matching task-bound single analysis cache')
    if not aligned and getattr(args, 'cached_prior_sources', ''):
        raise ValueError('cached priors require --budgeted-prompt full_flat_v5')
    config = SelectionConfig(args.indirect_budget, args.indirect_selector, tuple(args.joint_weights),
                             args.records_per_molecule, args.selection_seed)
    provider = runtime._resolve_provider_pool_config(args)
    extras = {content_hash(p.request_extra_body) for p in provider.providers}
    if len(extras) != 1:
        raise ValueError('all providers must use the same extra generation parameters')
    if args.annotation_db:
        wal = Path(args.annotation_db + '-wal')
        if wal.exists() and wal.stat().st_size:
            raise ValueError('use a frozen annotation SQLite snapshot, not a live WAL database')
    inputs = {}
    for task in args.tasks:
        spec = specs[task]
        paths = {'input': spec.input_jsonl, 'catalog': spec.family_manifest}
        if benchmark:
            paths.update(benchmark=Path(args.benchmark_manifest),
                train=Path(benchmark['tasks'][task]['train']['path']),
                heldout=Path(benchmark['tasks'][task]['heldout']['path']))
        if matched and args.evidence_setting != 'none':
            paths['direct_knn'] = Path(args.direct_knn_root) / task / 'morgan_knn' / 'all_train_unique_molecules' / f'{args.evaluation_subset}_predictions.jsonl'
        if args.evidence_setting != 'none':
            paths['index'] = spec.index
            source, catalog = source_paths(task)
            paths.update(source=source, source_catalog=catalog)
            index_manifest = _read(spec.index.with_name('manifest.json'))
            if index_manifest.get('max_record_examples') != 0:
                raise ValueError('budgeted selection requires an uncapped record index (max_record_examples=0)')
            if index_manifest.get('reference_pool') != runtime.REFERENCE_POOL or index_manifest.get('n_direct_heldout_records_after_filter') != 0:
                raise ValueError('requires current heldout-filtered benchmark index')
            runtime._heldout_filter_validation(task=task, split_scheme=args.split_scheme, index_manifest=index_manifest,
                heldout_path=paths.get('heldout', task_root(task, args.split_scheme) / 'heldout_molecule_condition_labels.jsonl'))
        inputs[task] = {k: {'path': str(p), 'sha256': sha256_file(p)} for k, p in paths.items()}
        if 'source' in paths and index_manifest.get('records_sha256') != inputs[task]['source']['sha256']:
            raise ValueError(f'{task}: index and current source hashes differ')
    manifest = {'version': VERSION, 'benchmark_profile': benchmark['profile'] if benchmark else '', 'setting': args.evidence_setting, 'selection': asdict(config),
        'prompt_version': getattr(args, 'budgeted_prompt', 'budgeted_v1'),
        'prompt_alignment': ('full_flat_v5_adapted.v1' if getattr(args, 'record_rendering', '') == 'full_fields'
                             else flat_prompt_alignment.VERSION) if aligned else None,
        'upstream_prompt_commit': flat_prompt_alignment.UPSTREAM if aligned else None,
        'prior_sources': priors, 'raw_tools_visible': not aligned,
        'feature_version': FEATURE_VERSION if _uses_assessments(args) else None,
        'assessment_required': _uses_assessments(args),
        'budget_grid': list(BUDGET_GRID), 'indirect_identity_policy': getattr(args, 'indirect_identity_policy', 'parent_disjoint'),
        'direct_identity_policy': 'matched_tdc_train_knn' if matched else ('scaffold_disjoint' if args.split_scheme == 'scaffold' else 'parent_disjoint'),
        'candidate_molecules': args.candidate_molecules, 'min_similarity': args.min_similarity,
        'direct_molecules': 3 if matched else args.candidate_molecules, 'direct_records_per_molecule': 1 if matched else args.direct_records_per_molecule,
        'direct_budget': args.direct_budget, 'direct_selector': 'matched_morgan_knn' if matched else args.direct_selector,
        'split_scheme': args.split_scheme, 'evaluation_subset': args.evaluation_subset,
        'visibility_mode': args.visibility_mode, 'uniform_quality': args.uniform_record_quality,
        'annotation_db': {'path': args.annotation_db, 'sha256': sha256_file(Path(args.annotation_db))} if args.annotation_db else None,
        'transfer': {'path': args.transfer_scores, 'sha256': sha256_file(Path(args.transfer_scores))} if args.transfer_scores else {'constant': 1},
        'skip_tool_prefetch': args.skip_tool_prefetch, 'tool_service_url': args.tool_service_url,
        'feature_inputs_only': args.feature_inputs_only, 'allow_native_features': args.allow_native_features,
        'model': args.model, 'generation': runtime._generation_settings(provider), 'max_tokens': args.max_tokens,
        'request_extra_body': provider.providers[0].request_extra_body,
        'inputs': inputs, 'code': _code_hashes(),
        'task_contracts': {task: content_hash(asdict(task_contract(args, task))) for task in args.tasks}}
    # Normalize tuple values before comparison with JSON checkpoints.
    manifest = json.loads(json.dumps(manifest))
    root = Path(args.output_root)
    _pin(root / 'experiment_manifest.json', manifest)
    args.manifest_hash = content_hash(manifest)
    if getattr(args, 'preparation_ready_file', ''):
        if not args.prepare_only:
            raise ValueError('preparation readiness is only published by prepare-only workers')
        write_json_atomic(Path(args.preparation_ready_file), {'manifest_hash':args.manifest_hash})
    if preparation_submit is not None:
        if not args.prepare_only or _uses_assessments(args):
            raise ValueError('shared preparation requires prepare-only simple baselines')
        preparation_submit(args, specs, manifest, config)
        return 0
    transfers = _transfer_map(args.transfer_scores)
    prepared = []
    for task in args.tasks:
        prepared.extend(_prepare_task(args, task, specs[task], manifest, config, transfers, preparation_cache))
        gc.collect()
    if _code_hashes() != manifest['code']:
        raise ValueError('implementation changed during preparation; use a fresh output root')
    if args.feature_inputs_only:
        summaries = [_read(root / 'feature_inputs' / f'{task}.summary.json') for task in args.tasks]
        write_json_atomic(root / 'feature_inputs' / 'summary.json', {'tasks':summaries, 'features_written':True,
            'source_rows':sum(s['rows'] for s in summaries), 'llm_candidate_rows':sum(s['llm_candidates'] for s in summaries),
            'manifest_hash':content_hash(manifest), 'scope':'exact retrieved candidates; no query labels'})
        write_json_atomic(root / 'execution_status.json', {'mode':'feature_inputs_only', 'phase':'complete', 'model_calls':0})
        return 0
    if args.annotation_db and sha256_file(Path(args.annotation_db)) != manifest['annotation_db']['sha256']:
        raise ValueError('annotation snapshot changed during preparation')
    write_json_atomic(root / 'execution_provider.json', asdict(provider))
    client = None if args.prepare_only else runtime._make_client(args, provider)
    try:
        failed = runtime._run_query_rounds(args, prepared, client, root, run_query=_run_query)
    finally:
        if client is not None and hasattr(client, 'close'):
            client.close()
    if not args.prepare_only:
        summarize(args, prepared, root)
    return 1 if failed else 0
