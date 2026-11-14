"""Offline verification and mean/SD for frozen naive and MolRAG repetitions."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import math
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory

from predict.baselines.molrag import score_saved_runs
from predict.prompts.paper_baseline import verify_batch
from predict.utils.json import read_json, sha256_file, write_json_atomic

TASKS = ('bbb_martins', 'bioavailability_ma', 'skin_reaction', 'ames', 'dili', 'carcinogens')
METHODS = ('deepseek_direct_fill', 'deepseek_both_fill', 'molrag')
CAVEAT = ('Historical provider differs; historical naive TDC Skin/Carcinogens omitted effort, '
          'all new runs use low. MolRAG retains the selected full-cohort failure penalty.')


def summarize(rows, expected):
    """Require all five runs, average tasks within run, then calculate sample SD."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ('benchmark', 'task', 'method'))
        groups[key].append(row)
        if row['n'] != expected.get(key) or any(not math.isfinite(row[m]) or not 0 <= row[m] <= 1
                                               for m in ('macro_f1', 'accuracy')):
            raise ValueError(f'Invalid cohort or metric: {key}')
    if set(groups) != set(expected) or any(sorted(r['repetition'] for r in group) != [1, 2, 3, 4, 5]
                                         for group in groups.values()):
        raise ValueError('Missing or duplicate task repetitions')
    rows = list(rows)
    for benchmark in ('conditioned', 'tdc'):
        for method in sorted({key[2] for key in expected}):
            for repetition in range(1, 6):
                group = [r for r in rows if (r['benchmark'], r['method'], r['repetition']) ==
                         (benchmark, method, repetition) and r['task'] in TASKS]
                if {r['task'] for r in group} != set(TASKS) or len(group) != 6:
                    raise ValueError('Six-task means require all six distinct tasks')
                row = dict(benchmark=benchmark, task='six_task_mean', method=method,
                           repetition=repetition, n=sum(r['n'] for r in group),
                           **{m: statistics.mean(r[m] for r in group) for m in ('macro_f1', 'accuracy')},
                           failures_counted_wrong=sum(r['failures_counted_wrong'] for r in group),
                           run_root='see individual tasks')
                rows.append(row)
                groups[benchmark, 'six_task_mean', method].append(row)
    summary = []
    for (benchmark, task, method), group in sorted(groups.items()):
        for scope, selected in [('historical_plus_four', group),
                                ('new_parcc_low_four', [r for r in group if r['repetition'] > 1])]:
            row = dict(benchmark=benchmark, task=task, method=method, scope=scope, n_runs=len(selected))
            for metric in ('macro_f1', 'accuracy'):
                values = [r[metric] for r in selected]
                row[metric+'_mean'], row[metric+'_sd'] = statistics.mean(values), statistics.stdev(values)
            summary.append(row)
    return rows, summary


def build(study, output, workers=4):
    if workers < 1:
        raise ValueError('Workers must be positive')
    if output.exists():
        raise ValueError('Use a new report directory; completed reports stay frozen')
    cells = read_json(study/'preflight.json')['cells']
    jobs = read_json(study/'jobs.json')
    expected = {(c['benchmark'], c['task'], c['method']): c['n'] for c in cells}
    methods = {c['method'] for c in cells}
    if (not methods or not methods <= set(METHODS) | {'deepseek_none'} or len(cells) != 12*len(methods)
            or set(expected) != {(b, t, m) for b in ('conditioned', 'tdc') for t in TASKS for m in methods}):
        raise ValueError('Expected both benchmarks and six tasks for each supported method')
    expected_jobs = 48*len(methods - {'molrag'}) + (4 if 'molrag' in methods else 0)
    if len(jobs) != expected_jobs or any(j['kind'] not in ('naive', 'molrag') for j in jobs):
        raise ValueError('Job count or kind differs from the selected study methods')
    if len({Path(j['root']).resolve() for j in jobs}) != len(jobs):
        raise ValueError('Repetitions must have distinct output roots')
    for cell in cells:
        if sha256_file(Path(cell['source_manifest'])) != cell['source_manifest_sha256']:
            raise ValueError('Historical source manifest changed')
    rows = []
    for cell in cells:
        if cell['method'] == 'molrag':
            continue
        selected = [j for j in jobs if j['kind'] == 'naive' and
                    all(j[k] == cell[k] for k in ('benchmark', 'task', 'method'))]
        source = Path(cell['source_manifest']).parent
        if sorted(j['repetition'] for j in selected) != [2, 3, 4, 5] or any(Path(j['source']) != source for j in selected):
            raise ValueError('Naive repetitions do not match the frozen source')
        for run in [dict(root=str(source), repetition=1), *selected]:
            if run['repetition'] != 1:
                migration = read_json(Path(run['root'])/'experiment_manifest.json')['migration']
                if (Path(migration['source_batch']).resolve() != source.resolve()
                        or migration['source_manifest_sha256'] != cell['source_manifest_sha256']):
                    raise ValueError('Naive run provenance differs from the study')
            rows.append(dict(benchmark=cell['benchmark'], task=cell['task'], method=cell['method'],
                             repetition=run['repetition'], n=cell['n'], failures_counted_wrong=0,
                             run_root=run['root']))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        reports = pool.map(verify_batch, [row['run_root'] for row in rows])
        for index, (row, report) in enumerate(zip(rows, reports, strict=True), 1):
            scores = report['tasks']
            if len(scores) != 1 or scores[0]['task'] != row['task'] or scores[0]['queries'] != row['n']:
                raise ValueError('Naive result cohort differs from the study')
            score = scores[0]
            row.update(macro_f1=score['macro_f1'], accuracy=score['accuracy'])
            if index % 5 == 0:
                print(f'Verified naive batches: {index}/{len(rows)}', flush=True)
    molrag = sorted((j for j in jobs if j['kind'] == 'molrag'), key=lambda j: j['repetition'])
    if 'molrag' in methods and ([j['repetition'] for j in molrag] != [2, 3, 4, 5] or len({j['source'] for j in molrag}) != 1):
        raise ValueError('MolRAG requires four repetitions of one frozen source')
    roots = [Path(molrag[0]['source']), *[Path(j['root']) for j in molrag]] if molrag else []
    if roots and (len({r.name for r in roots}) != 5 or any(Path(c['source_manifest']).parents[4] != roots[0]
                                                       for c in cells if c['method'] == 'molrag')):
        raise ValueError('MolRAG sources or run identities differ from the study')
    for root in roots[1:]:
        state = read_json(root/'execution_status.json')
        if state['done'] != state['total'] or state['total'] != sum(c['n'] for c in cells if c['method'] == 'molrag'):
            raise ValueError('MolRAG still has pending queries')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.llm-repeats-') as temporary:
        stage = Path(temporary)/'report'
        stage.mkdir()
        scored = score_saved_runs(roots, stage/'molrag_scored') if roots else {'cells': []}
        repetitions = {root.name: i for i, root in enumerate(roots, 1)}
        for score in scored['cells']:
            repetition = repetitions[score['run_name']]
            rows.append(dict(benchmark=score['benchmark'], task=score['task'], method='molrag',
                             repetition=repetition, n=score['n_total'], macro_f1=score['macro_f1'],
                             accuracy=score['accuracy'], failures_counted_wrong=score['n_failures_counted_wrong'],
                             run_root=str(roots[repetition-1])))
        rows, summary = summarize(rows, expected)
        status = dict(status='complete', verified_cells=len(cells)*5, errors=[])
        for name, data in [('per_run.csv', rows), ('summary.csv', summary)]:
            with (stage/name).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        write_json_atomic(stage/'aggregation_status.json', status)
        write_json_atomic(stage/'results.json', dict(status=status, runs=rows, summary=summary,
            sd='sample standard deviation, ddof=1', caveat=CAVEAT,
            sources={name: sha256_file(study/name) for name in ('preflight.json', 'jobs.json')},
            implementation_sha256=sha256_file(Path(__file__))))
        stage.rename(output)
    print(f"Complete: {status['verified_cells']} task scores and {len(summary)} summaries written to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True, help='Directory containing preflight.json and jobs.json')
    parser.add_argument('--output', type=Path, required=True, help='New offline report directory')
    parser.add_argument('--workers', type=int, default=4, help='Parallel read-only naive validators (default: 4)')
    args = parser.parse_args()
    build(args.study.resolve(), args.output.resolve(), args.workers)


if __name__ == '__main__':
    main()
