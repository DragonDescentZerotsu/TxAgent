"""Rebuild paper tables and optional figures from hash-bound saved scores."""
from collections import Counter, defaultdict
import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
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
        scope='Saved-result derivation only; no new prediction, performance or Ours provenance audit.')
    return tables, receipt


_BAR_STYLES = (
    ('None', 'N', '#DCE1E7', '#AAB2BD'),
    ('Naive · Direct', 'Naive D', '#BCD6ED', '#759CBA'),
    ('Naive · Direct + Indirect', 'Naive D+I', '#D9CFEB', '#A48DBD'),
    ('Ours · Direct', 'Ours D', '#F6DFD1', '#C99B81'),
    ('Ours · Direct + Indirect', 'Ours D+I', '#EEC8B4', '#B88064'),
)


def _paper_panel(tables, benchmark, config, cohorts, *, mean_only):
    """Render one published panel from the same validated rows as the CSV export."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    tasks = list(cohorts)
    backbones = list(config['curve_settings'])
    baselines = config['baseline_settings']
    task_labels = config['task_labels']
    points = {(p['group'], p['setting'], p['task']): float(p['macro_f1'])
              for p in tables['figure_points'] if p['benchmark'] == benchmark}
    means = {(p['group'], p['setting']): p['macro_f1']
             for p in tables['figure_means'] if p['benchmark'] == benchmark}
    rows = []
    for backbone in backbones:
        for stage, (setting, _, face, edge) in enumerate(_BAR_STYLES):
            key = ('None', 'Naive Direct', 'Naive Direct+Indirect', 'Ours Direct', 'Ours Direct+Indirect')[stage]
            rows.append(dict(group=backbone, setting=setting, kind='llm', stage=stage,
                face=face, edge=edge, mean=means[backbone, key],
                values={task: points[backbone, key, task] for task in tasks}))
    for b in baselines:
        mixed = b['pool'] == 'mixed'
        rows.append(dict(group=b['pool_label'], kind='ml', tick=b['tick'],
            face='white' if mixed else '#748396', edge='#748396', hatch='////' if mixed else None,
            mean=means[b['pool'], b['method']],
            values={task: points[b['pool'], b['method'], task] for task in tasks}))
    limits = {}
    for task in tasks:
        values = [r['values'][task] for r in rows]
        span = max(max(values)-min(values), .08)
        limits[task] = [max(0., math.floor((min(values)-.1*span)*20)/20),
                        min(1., math.ceil((max(values)+.22*span)*20)/20)]
    values = [r['mean'] for r in rows]
    limits['mean'] = [max(0., math.floor((min(values)-.025)*20)/20),
                      min(1., math.ceil((max(values)+.06)*20)/20)]
    panel_tasks = ['mean'] if mean_only else tasks
    columns = min(2, len(panel_tasks))
    size = (16, 9) if mean_only else (28.5+max(0, len(backbones)-2)*4, 18)
    fig, axes = plt.subplots(math.ceil(len(panel_tasks)/columns), columns, squeeze=False, figsize=size)
    ticks = list(dict.fromkeys(b['tick'] for b in baselines))
    backbone_step = 4.5
    ml_start = len(backbones)*backbone_step
    ml_x = [ml_start+.8*i for i in range(len(ticks))]
    for panel_index, (ax, task) in enumerate(zip(axes.flat, panel_tasks)):
        low, high = limits[task]
        x_ticks, x_labels, x_stages = [], [], []
        for i, backbone in enumerate(backbones):
            model_rows = [r for r in rows if r['kind'] == 'llm' and r['group'] == backbone]
            xs = [i*backbone_step + 1.6 + .8*(j-(len(model_rows)-1)/2)
                  for j in range(len(model_rows))]
            for x, row in zip(xs, model_rows):
                value = row['mean'] if mean_only else row['values'][task]
                ax.bar(x, value-low, bottom=low, width=.58, facecolor=row['face'],
                       edgecolor=row['edge'], linewidth=1.1, zorder=3)
            x_ticks.extend(xs)
            x_labels.extend(['N', 'D', 'D+I', 'D', 'D+I'][r['stage']] for r in model_rows)
            x_stages.extend(r['stage'] for r in model_rows)
            ax.text(mean(xs), 1.035, backbone, transform=ax.get_xaxis_transform(),
                    ha='center', fontsize=28, color='#566274')
            ax.plot([xs[0]-.3, xs[-1]+.3], [1.005, 1.005], transform=ax.get_xaxis_transform(),
                    color='#D4DAE2', lw=1.8, clip_on=False)
        for tick, x in zip(ticks, ml_x):
            peers = [r for r in rows if r['kind'] == 'ml' and r['tick'] == tick]
            for i, row in enumerate(peers):
                value = row['mean'] if mean_only else row['values'][task]
                offset = (i-(len(peers)-1)/2)*.35
                ax.bar(x+offset, value-low, bottom=low,
                       width=.31 if len(peers)>1 else .55, facecolor=row['face'],
                       edgecolor=row['edge'], linewidth=1.1, hatch=row['hatch'], zorder=3)
        ax.text(mean(ml_x), 1.035, 'ML baselines', transform=ax.get_xaxis_transform(),
                ha='center', fontsize=28, color='#566274')
        ax.plot([ml_x[0]-.4, ml_x[-1]+.4], [1.005, 1.005], transform=ax.get_xaxis_transform(),
                color='#D4DAE2', lw=1.8, clip_on=False)
        short_ticks = [t.replace('MM\nDirect', 'MM\nD').replace('MM\n+ Indirect', 'MM\n+I')
                       .replace('RASAR\n+indirect', 'R+I').replace('RASAR', 'R') for t in ticks]
        ax.set_xticks(x_ticks+ml_x, x_labels+short_ticks)
        ax.set_xlim(-.55, ml_x[-1]+.56)
        tick_values = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(low, high)
        ax.set_ylim(low, high)
        ax.set_yticks(sorted({low} | {float(t) for t in tick_values if low < t < high}))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='both', length=0, labelsize=26, pad=6, colors='#4D596B')
        ax.grid(axis='y', color='#E9EDF2', linewidth=.65)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#C7CFD9')
            ax.spines[side].set_linewidth(.65 * 1.15)
        if low > 0:
            for offset in (.02, .045):
                ax.plot([-.007, .007], [offset-.005, offset+.005], transform=ax.transAxes,
                        color='#657184', lw=.9, clip_on=False)
        ax.tick_params(axis='x', labelsize=26, pad=10)
        for i, label in enumerate(ax.get_xticklabels()):
            if i < len(x_stages) and x_stages[i] >= 3:
                label.set_color('#AA775D')
        if mean_only:
            ax.tick_params(axis='both', labelsize=28)
        else:
            ax.set_title(task_labels[task], loc='left', fontsize=38,
                         fontweight='normal', color='#263348', pad=64)
            ax.text(1, 1.32, f"n = {cohorts[task]['n']:,}",
                    transform=ax.transAxes, ha='right', fontsize=28, color='#657184')
        if len(backbones) > 2:
            for label in ax.get_xticklabels()[len(x_stages):]:
                label.set_fontsize(22)
        if not mean_only and panel_index + columns < len(panel_tasks):
            ax.tick_params(axis='x', labelbottom=False)
    fig.subplots_adjust(left=.065 if mean_only else .06,
                        right=.98,
                        top=.62 if mean_only else .69, bottom=.12 if mean_only else .07,
                        hspace=.95 if mean_only else .65, wspace=.23)
    header_left = .065 if mean_only else .06
    visible_settings = {r['setting'] for r in rows if r['kind'] == 'llm'}
    handles = [Patch(facecolor=c, edgecolor=e, linewidth=.7, label=name)
               for name, _, c, e in _BAR_STYLES if name in visible_settings]
    naive_handles = [h for h in handles if h.get_label() in {s[0] for s in _BAR_STYLES[:3]}]
    other_handles = [h for h in handles if h not in naive_handles]
    for group in dict.fromkeys(r['group'] for r in rows if r['kind'] == 'ml'):
        row = next(r for r in rows if r['group'] == group)
        label = 'ML · Main-task train' if mean_only and not row['hatch'] else group
        handle = Patch(facecolor=row['face'], edgecolor=row['edge'], hatch=row['hatch'], label=label)
        handles.append(handle)
        other_handles.append(handle)
    fig.text(header_left, .963, config['title'],
             fontsize=38 if mean_only else 46, fontweight='bold', color='#263348', va='top')
    if mean_only:
        for row_handles, y in ((naive_handles, .89), (other_handles, .835)):
            fig.legend(handles=row_handles, loc='upper left',
                       bbox_to_anchor=(header_left-.003, y), ncol=len(row_handles),
                       frameon=False, fontsize=20, handlelength=1.3, columnspacing=1.6)
    else:
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(header_left-.003, .925),
                   ncol=4, frameon=False, fontsize=28, handlelength=1.45, columnspacing=1.7)
    abbreviations = 'N = None   ·   D = Direct   ·   I = Indirect   ·   MM = MiniMol   ·   R = RASAR'
    fig.text(header_left, .73 if mean_only else .80, abbreviations,
             fontsize=22 if mean_only else 28, color='#748093' if mean_only else '#4D596B')
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Directory containing the compact paper result export')
    parser.add_argument('--output', type=Path, required=True, help='New derived report directory')
    parser.add_argument('--figures', action='store_true', help='Also render the three published PDFs (Matplotlib and pdflatex required)')
    args = parser.parse_args(argv)
    tables, receipt = build_tables(args.source)
    if args.output.resolve() == args.source.resolve() or args.source.resolve() in args.output.resolve().parents:
        raise ValueError('keep derived reports separate from frozen source files')
    if args.figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if shutil.which('pdflatex') is None:
            raise RuntimeError('--figures requires pdflatex with the standalone and graphicx packages')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        path = args.output/f'{name}.csv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    receipt['output_hashes'] = {f'{name}.csv': sha256_file(args.output/f'{name}.csv') for name in tables}
    if args.figures:
        display = json.loads((args.source/'display.json').read_text())
        cohorts = json.loads((args.source/'sources.json').read_text())['cohorts']
        for benchmark, label in (('conditioned', 'starling'), ('tdc', 'tdc')):
            (args.output/benchmark).mkdir()
            for mean_only in (False, True):
                fig = _paper_panel(tables, benchmark, display[benchmark], cohorts[benchmark], mean_only=mean_only)
                relative = f'{benchmark}/baseline_bars_mean.pdf' if mean_only else f'main_results_{label}.pdf'
                try:
                    fig.savefig(args.output/relative, dpi=200, facecolor='white')
                finally:
                    plt.close(fig)
                receipt['output_hashes'][relative] = sha256_file(args.output/relative)
        template = Path(__file__).with_name('paper_mean.tex')
        with tempfile.TemporaryDirectory(prefix='txagent-paper-mean-') as work:
            work = Path(work)
            shutil.copyfile(template, work/'mean_results.tex')
            for benchmark in display:
                (work/benchmark).mkdir()
                shutil.copyfile(args.output/benchmark/'baseline_bars_mean.pdf', work/benchmark/'baseline_bars_mean.pdf')
            result = subprocess.run(['pdflatex', '-halt-on-error', '-interaction=nonstopmode', 'mean_results.tex'],
                                    cwd=work, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError('Mean figure composition failed:\n' + result.stdout[-4000:])
            shutil.copyfile(work/'mean_results.pdf', args.output/'mean_results.pdf')
        receipt['output_hashes']['mean_results.pdf'] = sha256_file(args.output/'mean_results.pdf')
        receipt['figure_rendering'] = dict(matplotlib=matplotlib.__version__,
            renderer_sha256=sha256_file(Path(__file__)), composition_sha256=sha256_file(template),
            panels=4, task_bars=len(tables['figure_points']), mean_bars=len(tables['figure_means']))
    write_json_atomic(args.output/'receipt.json', receipt)
    print(json.dumps(receipt['cells']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
