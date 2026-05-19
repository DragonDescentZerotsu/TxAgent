"""Run MiniMol head hyperparameter sweeps for JSONL binary tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MINIMOL_ADMET_CONFIGS = [
    {"hidden_dim": 512, "depth": 3, "lr": 0.0001},
    {"hidden_dim": 512, "depth": 3, "lr": 0.0003},
    {"hidden_dim": 512, "depth": 4, "lr": 0.0003},
    {"hidden_dim": 512, "depth": 4, "lr": 0.0005},
    {"hidden_dim": 1024, "depth": 3, "lr": 0.0003},
    {"hidden_dim": 1024, "depth": 3, "lr": 0.0005},
    {"hidden_dim": 1024, "depth": 4, "lr": 0.0001},
    {"hidden_dim": 1024, "depth": 4, "lr": 0.0005},
    {"hidden_dim": 2048, "depth": 3, "lr": 0.0001},
    {"hidden_dim": 2048, "depth": 4, "lr": 0.0003},
    {"hidden_dim": 2048, "depth": 4, "lr": 0.0005},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("outputs/baselines/minimol_sweeps"), type=Path)
    parser.add_argument("--embedding-cache-dir", default=None, type=Path)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--conda", default="/data1/tianang/anaconda3/condabin/conda")
    parser.add_argument("--env-name", default="intern")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def config_id(config: dict[str, float | int]) -> str:
    lr = str(config["lr"]).replace(".", "p")
    return f"h{config['hidden_dim']}_d{config['depth']}_lr{lr}"


def run_config(args: argparse.Namespace, config: dict[str, float | int], gpu: str) -> dict:
    out_dir = args.output_root / args.task_name / config_id(config)
    metrics_path = out_dir / "metrics.json"
    log_path = out_dir / "run.log"
    if metrics_path.exists() and not args.force:
        with metrics_path.open() as f:
            metrics = json.load(f)
        metrics["sweep_status"] = "cached"
        metrics["sweep_output_dir"] = str(out_dir)
        return metrics

    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        args.conda,
        "run",
        "-n",
        args.env_name,
        "python",
        "-m",
        "baselines.minimol.run_bioavailability_ma",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(out_dir),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--depth",
        str(config["depth"]),
        "--lr",
        str(config["lr"]),
        "--epochs",
        str(args.epochs),
        "--ensemble-size",
        str(args.ensemble_size),
    ]
    if args.embedding_cache_dir is not None:
        cmd.extend(["--embedding-cache-dir", str(args.embedding_cache_dir)])
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        return {
            "sweep_status": "failed",
            "sweep_output_dir": str(out_dir),
            "sweep_config": config,
            "returncode": proc.returncode,
        }

    with metrics_path.open() as f:
        metrics = json.load(f)
    metrics["sweep_status"] = "ok"
    metrics["sweep_output_dir"] = str(out_dir)
    return metrics


def ranking_key(result: dict) -> tuple[float, float, float]:
    valid = result["valid_metrics"]
    return (
        float(valid["auroc"]),
        float(valid["macro_f1"]),
        -float(valid.get("threshold", 0.5)),
    )


def compact_result(result: dict) -> dict:
    args = result["args"]
    return {
        "status": result["sweep_status"],
        "output_dir": result["sweep_output_dir"],
        "hidden_dim": int(args["hidden_dim"]),
        "depth": int(args["depth"]),
        "lr": float(args["lr"]),
        "valid": result["valid_metrics"],
        "test": result["test_metrics"],
    }


def main() -> None:
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must not be empty")

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []
        for idx, config in enumerate(MINIMOL_ADMET_CONFIGS):
            futures.append(executor.submit(run_config, args, config, gpus[idx % len(gpus)]))
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result.get("sweep_status") == "failed":
                print(f"[sweep] failed {result['sweep_output_dir']}", file=sys.stderr)
            else:
                compact = compact_result(result)
                print(
                    "[sweep] done "
                    f"{compact['output_dir']} valid_auroc={compact['valid']['auroc']:.6f} "
                    f"test_auroc={compact['test']['auroc']:.6f}"
                )

    failed = [result for result in results if result.get("sweep_status") == "failed"]
    completed = [result for result in results if result.get("sweep_status") != "failed"]
    completed.sort(key=ranking_key, reverse=True)
    summary = {
        "task_name": args.task_name,
        "data_dir": str(args.data_dir),
        "selection_metric": "valid_metrics.auroc",
        "n_configs": len(MINIMOL_ADMET_CONFIGS),
        "n_completed": len(completed),
        "n_failed": len(failed),
        "best": compact_result(completed[0]) if completed else None,
        "results": [compact_result(result) for result in completed],
        "failed": failed,
    }
    summary_path = args.output_root / args.task_name / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"[sweep] wrote {summary_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
