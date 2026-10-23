"""Export paper appendix tables and figure values from hash-bound saved scores."""
from collections import Counter, defaultdict
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from predict.utils.json import sha256_file, write_json_atomic


def build_tables(source):
    source = Path(source)
    contract = json.loads((source/'sources.json').read_text())
    hashes = {name: sha256_file(source/name) for name in ('baselines.csv', 'ours_reported.json', 'display.json', 'ablations.csv')}
    if any(digest != contract['export_hashes'].get(name) for name, digest in hashes.items()):
        raise ValueError('paper result input hash changed')
    with (source/'baselines.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    ours = json.loads((source/'ours_reported.json').read_text())
    display = json.loads((source/'display.json').read_text())
    if set(display) != set(ours['curves']) or set(display) != set(contract['cohorts']):
        raise ValueError('paper benchmarks must match all source bindings')
    by_key = {(r['benchmark'], r['task'], r['method']): r for r in rows}
    if len(by_key) != len(rows) or Counter(r['family'] for r in rows) != {'baseline': 198, 'LLM': 108}:
        raise ValueError('paper requires 198 unique ML cells and 108 unique LLM cells')
    for r in rows:
        cohort = contract['cohorts'][r['benchmark']][r['task']]
        if int(r['n']) != cohort['n']:
            raise ValueError('paper cohort size changed')
        for metric in ('macro_f1', 'accuracy', 'auroc'):
            if metric == 'auroc' and r['family'] == 'LLM' and not r[metric]:
                continue
            if not math.isfinite(float(r[metric])) or not 0 <= float(r[metric]) <= 1:
                raise ValueError('invalid paper metric')
    points, reported, winners = [], [], []
    for benchmark, config in display.items():
        tasks = list(contract['cohorts'][benchmark])
        if len(tasks) != 6:
            raise ValueError('paper means require all six tasks')
        consumed = set()
        for group in config['baseline_settings']:
            members = group['members']
            if not members or len(set(members)) != len(members) or consumed.intersection(members):
                raise ValueError('ML display groups must have disjoint nonempty members')
            consumed.update(members)
            for task in tasks:
                candidates = [by_key[benchmark, task, m] for m in members]
                if any(r['family'] != 'baseline' or r['training_pool'] != group['pool'] for r in candidates):
                    raise ValueError('ML display group mixes training pools')
                # Original figure convention: per-task test maximum; ties follow member order.
                winner = max(candidates, key=lambda r: float(r['macro_f1']))
                points.append(dict(benchmark=benchmark, task=task, family='ML', group=group['pool'],
                                   setting=group['method'], macro_f1=winner['macro_f1']))
                winners.append(dict(benchmark=benchmark, task=task, group=group['method'],
                    selected_method=winner['method'], tied_methods=[r['method'] for r in candidates
                    if float(r['macro_f1']) == float(winner['macro_f1'])]))
        required = set(consumed)
        for backbone, methods in config['curve_settings'].items():
            if len(methods) != 3 or required.intersection(methods):
                raise ValueError('LLM display requires three distinct stages')
            required.update(methods)
            for setting, method in zip(('None', 'Naive Direct', 'Naive Direct+Indirect'), methods):
                for task in tasks:
                    row = by_key[benchmark, task, method]
                    if row['family'] != 'LLM':
                        raise ValueError('LLM display must use LLM cells')
                    points.append(dict(benchmark=benchmark, task=task, family='LLM', group=backbone,
                                       setting=setting, macro_f1=row['macro_f1']))
        if {(r['task'], r['method']) for r in rows if r['benchmark'] == benchmark} != {
                (t, m) for t in tasks for m in required}:
            raise ValueError('display would omit or invent paper results')
        curves = ours['curves'][benchmark]
        bindings = config['reported_curve_backbones']
        if (len(curves) != 3 or {c['label'] for c in curves} != set(bindings)
                or set(bindings.values()) != set(config['curve_settings'])):
            raise ValueError('reported curves must bind to the three backbones')
        for curve in curves:
            if not curve.get('source') or set(curve['scores']) != set(tasks):
                raise ValueError('reported scores require a source and all tasks')
            for task, values in curve['scores'].items():
                if (len(values) != 3 or values[0] is not None or any(
                        v is None or not math.isfinite(v) or not 0 <= v <= 1 for v in values[1:])):
                    raise ValueError('reported None must stay shared; Direct stages must be complete')
                for setting, value in zip(('Ours Direct', 'Ours Direct+Indirect'), values[1:]):
                    point = dict(benchmark=benchmark, task=task, family='Ours reported',
                                 group=bindings[curve['label']], setting=setting, macro_f1=value)
                    points.append(point)
                    reported.append(dict(point, source=curve['source']))
    groups = defaultdict(list)
    for p in points:
        groups[p['benchmark'], p['family'], p['group'], p['setting']].append(p)
    means = []
    for key, items in groups.items():
        if len(items) != 6 or {p['task'] for p in items} != set(contract['cohorts'][key[0]]):
            raise ValueError('refusing a partial-task or duplicate-task mean')
        means.append(dict(zip(('benchmark', 'family', 'group', 'setting'), key),
                          macro_f1=mean(float(p['macro_f1']) for p in items), n_tasks=6))
    with (source/'ablations.csv').open(newline='') as handle:
        ablations = list(csv.DictReader(handle))
    ablation_groups = defaultdict(list)
    for row in ablations:
        value = float(row['macro_f1'])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('invalid ablation score')
        ablation_groups[row['condition']].append(row)
    if len(ablations) != 30 or len(ablation_groups) != 5 or 'Control' not in ablation_groups:
        raise ValueError('requires the five source-backed ablation conditions')
    for items in ablation_groups.values():
        if len(items) != 6 or {r['task'] for r in items} != set(contract['cohorts']['conditioned']):
            raise ValueError('ablation means require six unique tasks')
    control = mean(float(r['macro_f1']) for r in ablation_groups['Control'])
    ablation_means = [dict(condition=k, macro_f1=mean(float(r['macro_f1']) for r in items),
                          change=mean(float(r['macro_f1']) for r in items)-control, n_tasks=6)
                      for k, items in ablation_groups.items()]
    tables = {'ml_appendix': [r for r in rows if r['family'] == 'baseline'],
              'llm_appendix': [r for r in rows if r['family'] == 'LLM'], 'ours_reported': reported,
              'figure_points': points, 'figure_means': means,
              'component_ablation': ablations, 'component_ablation_means': ablation_means}
    receipt = dict(source_hashes=hashes, sources_sha256=sha256_file(source/'sources.json'),
        cells={k: len(v) for k,v in tables.items()}, ml_group_selections=winners,
        ablation_source=contract['ablation_source'],
        aggregation='Equal six-task means; ML uses per-task group test maxima, not validation selection.',
        scope='Saved table derivation only; no new prediction, performance or Ours provenance audit.')
    return tables, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Directory containing the compact paper result export')
    parser.add_argument('--output', type=Path, required=True, help='New derived report directory')
    args = parser.parse_args(argv)
    tables, receipt = build_tables(args.source)
    if args.output.resolve() == args.source.resolve() or args.source.resolve() in args.output.resolve().parents:
        raise ValueError('keep derived reports separate from frozen source files')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        path = args.output/f'{name}.csv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    receipt['output_hashes'] = {f'{name}.csv': sha256_file(args.output/f'{name}.csv') for name in tables}
    write_json_atomic(args.output/'receipt.json', receipt)
    print(json.dumps(receipt['cells']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
