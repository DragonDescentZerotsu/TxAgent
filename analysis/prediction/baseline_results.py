"""Build comparable ML seed means without changing the frozen paper export."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from predict.baselines.repeated_runs import statistics
from predict.utils.json import atomic_output_path, sha256_file, write_json_atomic

METRICS = ("macro_f1", "auroc", "accuracy")
KNN = {"minimol_knn_all": ("minimol_knn", "all"), "minimol_knn_condition": ("minimol_knn", "condition"),
       "morgan_knn_all": ("morgan_knn", "all"), "morgan_knn_condition": ("morgan_knn", "condition"),
       "D2": ("minimol_knn", "all"), "D3": ("minimol_knn", "condition"),
       "D4": ("morgan_knn", "all"), "D5": ("morgan_knn", "condition")}


def build(root, source_file, output):
    config = json.loads(source_file.read_text())
    paper = root / config["historical_export"]
    binding = json.loads((paper / "sources.json").read_text())
    if sha256_file(paper / "baselines.csv") != binding["export_hashes"]["baselines.csv"]:
        raise ValueError("Historical paper score export changed")
    historical = list(csv.DictReader((paper / "baselines.csv").open()))
    old = {(r["training_pool"], r["task"], r["method"]): r for r in historical if r["family"] == "baseline"}
    scores, seeds, receipts = [], [], {}
    for relative, expected in config["repeated_sources"].items():
        source = root / relative
        verified = json.loads((source / "verification.json").read_text())
        rows = list(csv.DictReader((source / "scores.csv").open()))
        members = list(csv.DictReader((source / "per_seed.csv").open()))
        if not verified["complete"] or verified["verified_cells"] != expected or len(rows) != expected or len(members) != expected * 5:
            raise ValueError(f"Incomplete repeated-run source: {source}")
        grouped = defaultdict(list)
        for row in members:
            key = tuple(row[k] for k in ("model", "training_pool", "task", "setting"))
            grouped[key].append(row)
            seeds.append({k: row[k] for k in ("model", "training_pool", "task", "setting", "seed", *METRICS)})
        for row in rows:
            key = tuple(row[k] for k in ("model", "training_pool", "task", "setting"))
            group = grouped[key]
            if sorted(int(m["seed"]) for m in group) != [1, 2, 3, 4, 5]:
                raise ValueError(f"Missing or duplicate seeds: {key}")
            for metric in METRICS:
                actual = statistics([float(m[metric]) for m in group])
                if any(not np.isclose(v, float(row[f"{metric}_{k}"]), rtol=1e-10, atol=1e-12)
                       for k, v in actual.items() if k != "n"):
                    raise ValueError(f"Repeated summary disagrees with members: {key}")
            record = dict(row, aggregation="independent_seed_mean", source=str(source), legacy_ensemble_macro_f1="")
            model, pool, task, setting = key
            prefix = "mixed_" if pool == "mixed" else "tdc_only_" if pool == "tdc_only" else ""
            if model == "minimol":
                method = ("D1" if pool == "mixed" else "minimol_head") if setting == "frozen" else prefix + setting
                record["legacy_ensemble_macro_f1"] = old[pool, task, method]["macro_f1"]
            elif model == "rasar":
                method = ("mixed_" if pool == "mixed" else "") + "rf_" + setting.replace("_all", "")
                record["legacy_ensemble_macro_f1"] = old[pool, task, method]["macro_f1"]
            scores.append(record)
        receipts[relative] = {f: sha256_file(source / f) for f in ("scores.csv", "per_seed.csv", "verification.json")}
    single = [r for r in historical if r["method"] in KNN]
    if len(single) != config["expected_single_cells"]:
        raise ValueError("Incomplete deterministic KNN reference")
    for row in single:
        model, setting = KNN[row["method"]]
        base = dict(model=model, training_pool=row["training_pool"], task=row["task"], setting=setting)
        record = dict(base, n_seeds=1, precision="deterministic", aggregation="single_deterministic",
                      source=str(paper / "baselines.csv"), legacy_ensemble_macro_f1="")
        for metric in METRICS:
            record.update({f"{metric}_{k}": row[metric] if k == "mean" else ""
                           for k in ("mean", "sd", "ci95_low", "ci95_high")})
        scores.append(record)
        seeds.append(dict(base, seed="", **{k: row[k] for k in METRICS}))
    keys = [tuple(r[k] for k in ("model", "training_pool", "task", "setting")) for r in scores]
    if len(set(keys)) != len(keys):
        raise ValueError("Result sources contain duplicate cells")
    # Aggregate tasks within seed, then compute uncertainty across seeds.
    groups = defaultdict(lambda: defaultdict(list))
    for row in seeds:
        groups[row["model"], row["training_pool"], row["setting"]][row["seed"]].append(row)
    means = []
    for (model, pool, setting), members in sorted(groups.items()):
        if any(len(rows) != 6 or len({r["task"] for r in rows}) != 6 for rows in members.values()):
            raise ValueError("Six-task means require six distinct tasks for every seed")
        record = dict(model=model, training_pool=pool, setting=setting, n_seeds=len(members))
        for metric in METRICS:
            values = [float(np.mean([float(r[metric]) for r in rows])) for rows in members.values()]
            result = statistics(values) if len(values) > 1 else dict(mean=values[0], sd="", ci95_low="", ci95_high="")
            record.update({f"{metric}_{k}": v for k, v in result.items() if k != "n"})
        means.append(record)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("scores.csv", scores), ("per_seed.csv", seeds), ("six_task_means.csv", means)):
        with atomic_output_path(output / name) as temporary:
            with temporary.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Comparable ML baseline means", "", "Five independent seeds: mean ± sample SD [95% t CI]. KNN is deterministic; its SD/CI are unavailable, not zero.",
             "Six-task means weight tasks equally within each seed. No per-task test-max model selection. Frozen historical ensemble scores remain in scores.csv as a separate column.",
             "New encoders use a fixed MiniMol-derived auxiliary lambda; historical MiniMol and RASAR select candidates separately within each seed.",
             "", "| Model | Pool | Setting | Runs | Six-task Macro-F1 |", "| --- | --- | --- | ---: | --- |"]
    for row in means:
        value = f"{row['macro_f1_mean']:.4f}"
        if row["n_seeds"] > 1:
            value += f" ± {row['macro_f1_sd']:.4f} [{row['macro_f1_ci95_low']:.4f}, {row['macro_f1_ci95_high']:.4f}]"
        lines.append(f"| {row['model']} | {row['training_pool']} | {row['setting']} | {row['n_seeds']} | {value} |")
    (output / "MAIN_TABLE.md").write_text("\n".join(lines) + "\n")
    export = config.get("comparison_export")
    if export:
        # One compact repository record; detailed predictions stay in run artifacts.
        with atomic_output_path(root / export) as temporary:
            with temporary.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=[k for k in scores[0] if k != "source"], extrasaction="ignore")
                writer.writeheader(); writer.writerows(scores)
    write_json_atomic(output / "verification.json", dict(complete=True, cells=len(scores), repeated_cells=len(scores)-len(single),
        deterministic_cells=len(single), evaluated_members=len(seeds), source_file=str(source_file),
        source_file_sha256=sha256_file(source_file), historical_scores_sha256=sha256_file(paper / "baselines.csv"),
        comparison_export_sha256=sha256_file(root / export) if export else None, sources=receipts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("experiments/baselines/results.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads(args.sources.read_text())
    build(root, args.sources.resolve(), args.output or root / config["comparison_output"])


if __name__ == "__main__":
    main()
