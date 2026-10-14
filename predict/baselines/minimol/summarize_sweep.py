"""Summarize a completed MiniMol sweep using validation AUROC, F1 and threshold."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def summarize(task_name: str, data_dir: Path, output_root: Path, status_file: Path) -> dict:
    completed, failed = [], []
    with status_file.open(newline="") as handle:
        statuses = list(csv.reader(handle, delimiter="\t"))
    for run_id, status, returncode in statuses:
        output = output_root / task_name / run_id
        if status == "failed":
            failed.append(dict(sweep_status=status, sweep_output_dir=str(output), returncode=int(returncode)))
            continue
        metrics = json.loads((output / "metrics.json").read_text())
        settings = metrics["args"]
        completed.append(dict(status=status, output_dir=str(output), hidden_dim=int(settings["hidden_dim"]),
                              depth=int(settings["depth"]), lr=float(settings["lr"]),
                              valid=metrics["valid_metrics"], test=metrics["test_metrics"]))
    completed.sort(key=lambda row: (float(row["valid"]["auroc"]), float(row["valid"]["macro_f1"]),
                                   -float(row["valid"].get("threshold", 0.5))), reverse=True)
    summary = dict(task_name=task_name, data_dir=str(data_dir), selection_metric="valid_metrics.auroc",
                   n_configs=len(statuses), n_completed=len(completed), n_failed=len(failed),
                   best=completed[0] if completed else None, results=completed, failed=failed)
    path = output_root / task_name / "sweep_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.task_name, args.data_dir, args.output_root, args.status_file)
    print(json.dumps({key: summary[key] for key in ("n_configs", "n_completed", "n_failed")}))
    if summary["n_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
