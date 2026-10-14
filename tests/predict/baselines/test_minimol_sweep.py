import json
import os
from pathlib import Path
import subprocess


RECIPE = Path(__file__).resolve().parents[3] / "predict/baselines/minimol/run_direct_gpu_sweep.sh"


def test_bash_sweep_serializes_each_gpu_skips_cache_and_reports_failures(tmp_path):
    interpreter = tmp_path / "fake python"
    interpreter.write_text('''#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
if sys.argv[2].endswith("summarize_sweep"):
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
args = dict(zip(sys.argv[3::2], sys.argv[4::2]))
root = Path(args["--output-dir"])
lock = root.parents[1] / ("gpu" + os.environ["CUDA_VISIBLE_DEVICES"] + ".lock")
with lock.open("x"):
    try:
        time.sleep(.02)
        if root.name == os.environ.get("FAIL_CONFIG"):
            raise SystemExit(3)
        settings = {name: args["--" + name.replace("_", "-")] for name in ("hidden_dim", "depth", "lr")}
        (root / "metrics.json").write_text(json.dumps(dict(args=settings,
            valid_metrics=dict(auroc=.8, macro_f1=.7, threshold=.5), test_metrics=dict(auroc=.6))))
    finally:
        lock.unlink()
''')
    interpreter.chmod(0o755)
    output = tmp_path / "output"
    command = ["bash", str(RECIPE), "task", str(tmp_path / "data space"), str(tmp_path / "cache"), str(output), "0,1"]
    environment = dict(os.environ, PYTHON_BIN=str(interpreter))
    dry = subprocess.run(command, env=dict(environment, DRY_RUN="1"), capture_output=True, text=True, check=True)
    assert len(dry.stdout.splitlines()) == 11 and not output.exists()
    subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
    summary = json.loads((output / "task/sweep_summary.json").read_text())
    assert (summary["n_configs"], summary["n_completed"], summary["n_failed"]) == (11, 11, 0)
    subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
    assert {row["status"] for row in json.loads((output / "task/sweep_summary.json").read_text())["results"]} == {"cached"}
    failed = subprocess.run(command, env=dict(environment, FORCE="1", FAIL_CONFIG="h512_d3_lr0p0001"),
                            capture_output=True, text=True)
    assert failed.returncode == 1
    summary = json.loads((output / "task/sweep_summary.json").read_text())
    assert (summary["n_completed"], summary["n_failed"]) == (10, 1)
    assert summary["failed"][0]["returncode"] == 3
