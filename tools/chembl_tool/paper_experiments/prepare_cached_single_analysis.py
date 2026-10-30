"""Prepare frozen single-molecule analyses for TDC through the shared query pool.

No neighbor retrieval or final prediction. Native single-analysis prompts, tool
prefetch, validation, provider admission and retries reuse the existing runtime.
"""
import argparse
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

from tools.chembl_tool.common.budgeted_evidence import content_hash
from tools.chembl_tool.common.conditioned_query_prior import build_query_prior_messages
from tools.chembl_tool.common.flat_prompt_alignment import project_prior
from tools.chembl_tool.common.identity_blind import prepare_reasoning_retrieval
from tools.chembl_tool.common.json_utils import read_jsonl, sha256_file, write_json_atomic
from tools.chembl_tool.common.reasoning_payload import llm_query_payload
from tools.chembl_tool.common.reasoning_validation import call_with_json_validation, structured_response_is_valid
from tools.chembl_tool.common.tdc_benchmark import PROFILE, task_contract
from tools.chembl_tool.paper_experiments import budgeted_evidence as budgeted
from tools.chembl_tool.paper_experiments import run_conditioned_assay_progressive_curve as runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--provider-pool-config', required=True)
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-V4-Flash-0731')
    parser.add_argument('--tasks', nargs='+', default=['bbb_martins', 'bioavailability_ma', 'skin_reaction', 'ames', 'dili', 'carcinogens'])
    parser.add_argument('--parallelism', type=int, default=256)
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--tool-service-url', default='http://127.0.0.1:8765')
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only; does not publish full caches.')
    args = parser.parse_args()
    runtime.load_env_file(Path('.env'))
    args.prepare_only, args.parallelism_per_task = False, 0
    args.max_stage_requeues, args.retry_delay_s = 3, 15
    args.timeout_s, args.transport_max_retries, args.retry_race_width = 900, 0, 1
    provider = runtime._resolve_provider_pool_config(args)
    root = args.output_root
    contracts = {t: task_contract(runtime._task_contract(t), PROFILE) for t in args.tasks}
    paths = {t: args.data_root/t/'scaffold/test.jsonl' for t in args.tasks}
    rows, queries = {}, []
    for task, path in paths.items():
        # Labels never enter the in-memory query rows passed to the model.
        unique = {(r['drug'], r.get('condition_group', '')): None for r in read_jsonl(path)}
        rows[task] = list(unique)
        queries.extend(runtime.PreparedQuery(task, i, root/task/f'query_{i:05d}')
            for i in range(min(len(unique), args.limit) if args.limit else len(unique)))
    manifest = dict(version='single_analysis_preparation.v1', benchmark_profile=PROFILE,
        model=args.model, generation=runtime._generation_settings(provider), max_tokens=args.max_tokens,
        request_extra_body=provider.providers[0].request_extra_body,
        inputs={t: dict(path=str(p.resolve()), sha256=sha256_file(p)) for t, p in paths.items()},
        task_contracts={t: content_hash(asdict(c)) for t, c in contracts.items()},
        code={str(p): sha256_file(p) for p in [Path(__file__), Path(runtime.__file__),
            Path('tools/chembl_tool/common/tdc_benchmark.py'), Path('tools/chembl_tool/common/conditioned_query_prior.py')]})
    budgeted._pin(root/'manifest.json', manifest)
    binding = content_hash(manifest)

    def run(_args, query, client):
        smiles, condition = rows[query.task][query.index]
        path = query.query_dir/'analysis.json'
        if path.exists():
            saved = budgeted._read(path)
            if saved['manifest_hash'] != binding or saved['query_hash'] != content_hash([smiles, condition]):
                raise ValueError('cached single analysis binding changed')
            project_prior(saved['analysis'])
            return dict(task=query.task, index=query.index, status='ok')
        retrieval_path = query.query_dir/'tools.json'
        if retrieval_path.exists():
            retrieval = budgeted._read(retrieval_path)
            if retrieval['query']['input_smiles'] != smiles:
                raise ValueError('prior tool cache has a different molecule')
        else:
            retrieval = prepare_reasoning_retrieval(
                {'query': {'input_smiles': smiles, 'canonical_smiles': smiles}, 'groups': [],
                 'experiment': {'mode': 'none', 'source': 'starling'}},
                runtime.ToolServiceClient(args.tool_service_url, timeout_s=args.timeout_s),
                identity_blind=False, harness_prefetch_tools=True)
            write_json_atomic(retrieval_path, retrieval)
        properties = retrieval['query']['prefetched_molecule_properties']
        if properties.get('status') != 'ok' or not runtime.tool_functional_group_tree(properties):
            raise ValueError('single analysis requires successful property tools including functional-group tree')
        payload = llm_query_payload(retrieval['query'])
        if query.task in ('bbb_martins', 'bioavailability_ma', 'skin_reaction'):
            config = import_module(f'tools.chembl_tool.tasks.{query.task}.run_reasoning_pipeline')
            response = config._reason_single_molecule(client, payload)['llm']
        else:
            messages = build_query_prior_messages(contracts[query.task], 'single', payload)
            response = call_with_json_validation(client.chat_json, messages,
                required_fields=('confidence', 'reasoning_summary'), branch_name='single-molecule')
        write_json_atomic(query.query_dir/'response.json', response)
        if not structured_response_is_valid(response):
            raise ValueError('invalid single analysis response')
        analysis = project_prior(response['content'])
        write_json_atomic(path, dict(manifest_hash=binding, query_hash=content_hash([smiles, condition]),
            query_smiles=smiles, condition_group=condition, analysis=analysis))
        return dict(task=query.task, index=query.index, status='ok')

    client = runtime._make_client(args, provider)
    try:
        failed = runtime._run_query_rounds(args, queries, client, root, run_query=run)
    finally:
        client.close()
    if failed or args.limit:
        return int(bool(failed))
    mapping = {}
    for task in args.tasks:
        path = root/task/'single_analysis.json'
        write_json_atomic(path, dict(version='single_analysis_cache.v1', task=task,
            benchmark_profile=PROFILE, task_contract_hash=manifest['task_contracts'][task],
            input=manifest['inputs'][task], preparation_manifest_hash=binding,
            rows=[budgeted._read(q.query_dir/'analysis.json') for q in queries if q.task == task]))
        mapping[task] = str(path.resolve())
    write_json_atomic(root/'prior_sources.json', mapping)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
