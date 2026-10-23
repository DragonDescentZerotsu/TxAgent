"""The paper report must preserve complete cohorts and its declared display rule."""
import json
import shutil
from pathlib import Path

import pytest

from analysis.prediction.paper_results import build_tables, main
from predict.utils.json import sha256_file

SOURCE = Path(__file__).resolve().parents[2]/'docs/paper'


def test_export_keeps_appendices_and_per_task_group_maxima(tmp_path):
    tables, receipt = build_tables(SOURCE)
    assert receipt['cells'] == dict(ml_appendix=198, llm_appendix=108, ours_reported=72,
                                   figure_points=270, figure_means=45,
                                   component_ablation=30, component_ablation_means=5)
    # KNN's paper mean is mean(per-task maxima), not max(per-method means).
    means = [r for r in tables['figure_means'] if r['setting'] == 'knn']
    assert [round(r['macro_f1'], 4) for r in means] == [.6025, .7036]
    assert all(r['n_tasks'] == 6 for r in tables['figure_means'])
    assert not any(r['setting'] == 'None' for r in tables['ours_reported'])
    assert all(r['auroc'] == '' for r in tables['llm_appendix'])
    assert [round(r['macro_f1'], 3) for r in tables['component_ablation_means']] == [.721, .704, .714, .711, .714]
    out = tmp_path/'report'
    assert main(['--source', str(SOURCE), '--output', str(out)]) == 0
    assert len(list(out.glob('*.csv'))) == 7
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        main(['--source', str(SOURCE), '--output', str(out)])
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


@pytest.mark.parametrize('change', ['hash', 'cohort', 'incomplete', 'shared_none', 'mixed_pool'])
def test_rejects_unbound_incomplete_or_misgrouped_results(tmp_path, change):
    root = tmp_path/'source'; shutil.copytree(SOURCE, root)
    if change == 'cohort':
        path = root/'sources.json'; doc = json.loads(path.read_text())
        doc['cohorts']['tdc']['bbb_martins']['n'] += 1
    elif change in ('incomplete', 'shared_none'):
        path = root/'ours_reported.json'; doc = json.loads(path.read_text())
        scores = doc['curves']['tdc'][0]['scores']
        if change == 'incomplete': del scores['bbb_martins']
        else: scores['bbb_martins'][0] = .5
    else:
        path = root/'display.json'; doc = json.loads(path.read_text())
        doc['tdc']['baseline_settings'][0]['pool'] = 'mixed'
    path.write_text(json.dumps(doc))
    if change != 'hash' and path.name != 'sources.json':
        manifest = json.loads((root/'sources.json').read_text())
        manifest['export_hashes'][path.name] = sha256_file(path)
        (root/'sources.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        build_tables(root)


@pytest.mark.parametrize('benchmark,mean_only', [('conditioned', False), ('tdc', False),
                                               ('conditioned', True), ('tdc', True)])
def test_published_panels_plot_all_and_only_exported_scores(benchmark, mean_only):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analysis.prediction.paper_results import _paper_panel

    tables, _ = build_tables(SOURCE)
    config = json.loads((SOURCE/'display.json').read_text())[benchmark]
    cohorts = json.loads((SOURCE/'sources.json').read_text())['cohorts'][benchmark]
    fig = _paper_panel(tables, benchmark, config, cohorts, mean_only=mean_only)
    try:
        assert len(fig.axes) == (1 if mean_only else 6)
        source = tables['figure_means' if mean_only else 'figure_points']
        for ax, task in zip(fig.axes, ['mean'] if mean_only else cohorts):
            expected = [float(r['macro_f1']) for r in source if r['benchmark'] == benchmark
                        and (mean_only or r['task'] == task)]
            actual = [p.get_y()+p.get_height() for p in ax.patches]
            assert len(actual) == (20 if benchmark == 'conditioned' else 25)
            assert sorted(actual) == pytest.approx(sorted(expected), abs=1e-12)
            if not mean_only:
                assert ax.get_title(loc='left') == config['task_labels'][task]
        assert fig.get_size_inches() == pytest.approx([16, 9] if mean_only else [32.5, 18])
        assert fig.texts[0].get_text() == config['title']
    finally:
        plt.close(fig)
