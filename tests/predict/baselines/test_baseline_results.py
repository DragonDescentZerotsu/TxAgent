import csv
import json

import pytest

from analysis.prediction.baseline_results import METRICS, build
from predict.baselines.repeated_runs import statistics
from predict.utils.json import sha256_file


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_comparison_preserves_seed_covariance_and_single_run_uncertainty(tmp_path):
    source, paper = tmp_path / "repeated", tmp_path / "paper"
    source.mkdir()
    paper.mkdir()
    members, summaries, historical = [], [], []
    for task in range(6):
        base = dict(model="new_encoder", training_pool="pool", task=str(task), setting="frozen")
        # Opposite seed effects cancel in the task mean, unlike mean task SD.
        values = [.7 + (seed-3) * .02 * (-1 if task % 2 else 1) for seed in range(1, 6)]
        members.extend(dict(base, seed=seed, **{m: value for m in METRICS}) for seed, value in enumerate(values, 1))
        record = dict(base, n_seeds=5, precision="fp32")
        for metric in METRICS:
            record.update({f"{metric}_{k}": v for k, v in statistics(values).items() if k != "n"})
        summaries.append(record)
        historical.append(dict(family="baseline", training_pool="pool", task=str(task),
                               method="morgan_knn_all", **{m: .6 for m in METRICS}))
    write_csv(source / "scores.csv", summaries)
    write_csv(source / "per_seed.csv", members)
    (source / "verification.json").write_text(json.dumps(dict(complete=True, verified_cells=6)))
    write_csv(paper / "baselines.csv", historical)
    (paper / "sources.json").write_text(json.dumps(dict(export_hashes={"baselines.csv": sha256_file(paper / "baselines.csv")})))
    config = tmp_path / "sources.json"
    config.write_text(json.dumps(dict(historical_export="paper", repeated_sources={"repeated": 6}, expected_single_cells=6,
                                     comparison_export="baseline_means.csv")))
    output = tmp_path / "comparison"
    build(tmp_path, config, output)
    means = list(csv.DictReader((output / "six_task_means.csv").open()))
    neural = next(r for r in means if r["model"] == "new_encoder")
    knn = next(r for r in means if r["model"] == "morgan_knn")
    assert float(neural["macro_f1_mean"]) == pytest.approx(.7)
    assert float(neural["macro_f1_sd"]) == pytest.approx(0, abs=1e-15)
    assert knn["macro_f1_sd"] == knn["macro_f1_ci95_low"] == knn["macro_f1_ci95_high"] == ""
    assert json.loads((output / "verification.json").read_text())["cells"] == 12
    exported = list(csv.DictReader((tmp_path / "baseline_means.csv").open()))
    assert len(exported) == 12 and "source" not in exported[0]
    assert sha256_file(paper / "baselines.csv") == json.loads((paper / "sources.json").read_text())["export_hashes"]["baselines.csv"]
    summaries[0]["macro_f1_mean"] = .99
    write_csv(source / "scores.csv", summaries)
    with pytest.raises(ValueError, match="disagrees with members"):
        build(tmp_path, config, output)
