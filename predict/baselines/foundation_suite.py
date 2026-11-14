"""Frozen paper inputs, GPU scheduling, and verification for foundation baselines."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from predict.baselines.foundation_models import MODELS, prepare_features
from predict.baselines.minimol.run_bioavailability_ma import choose_threshold, evaluate_metrics
from predict.baselines.minimol.run_multitask import ensemble_bce
from predict.utils.json import atomic_output_path, sha256_file, write_json_atomic


def prepare(args):
    source = json.loads(args.sources.read_text())
    runtime = json.loads((args.runtime_root / "manifest.json").read_text())
    if set(runtime["models"]) != set(MODELS):
        raise ValueError("All seven pinned model releases must be present")
    if sha256_file(args.source_root / source["source_table"]) != source["source_table_sha256"]:
        raise ValueError("Paper source table changed")
    cells = []
    for entry in source["datasets"]:
        original = args.source_root / entry["prepared"]
        for file, digest in (("manifest.json", entry["prepared_sha256"]), ("data.pt", entry["data_sha256"])):
            if sha256_file(original / file) != digest:
                raise ValueError(f"Changed prepared paper input: {original / file}")
        payload = torch.load(original / "data.pt", map_location="cpu", weights_only=False)
        for split in ("train", "valid", "test"):
            path = args.source_root / entry["data_dir"] / f"{split}.jsonl"
            hashes = [v for p, v in entry["inputs"].items() if Path(p).name == path.name]
            if hashes != [sha256_file(path)]:
                raise ValueError(f"Changed frozen split: {path}")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            stored = payload["splits"][split]
            if len(rows) != len(stored["indices"]):
                raise ValueError(f"Split size mismatch: {path}")
            for row, index, label in zip(rows, stored["indices"], stored["labels"]):
                if row["drug"] != payload["smiles"][int(index)] or row["Y"] != int(label):
                    raise ValueError(f"Prepared target order/identity mismatch: {path}")
        target = args.output / "inputs" / entry["training_pool"] / entry["task"]
        target.mkdir(parents=True, exist_ok=True)
        for file in ("data.pt", "manifest.json"):
            if (target / file).exists():
                if sha256_file(target / file) != sha256_file(original / file):
                    raise ValueError(f"Existing inputs changed: {target}")
            else:
                shutil.copyfile(original / file, target / file)
        task = entry["task"]
        cells.append(dict(**entry, input=str(target.relative_to(args.output)),
                          hidden_dim=2048 if task == "bbb_martins" else 512,
                          depth=4 if task == "dili" else 3,
                          lr=1e-4 if task in ("bbb_martins", "ames") else 5e-4 if task == "dili" else 3e-4))
    manifest = dict(schema="foundation_paper_batch.v1", run_name=args.run_name,
                    release=source["release"], sources_sha256=sha256_file(args.sources),
                    models=runtime["models"], cells=cells, settings=["frozen", "finetuned", "indirect"],
                    seeds=[1, 2, 3, 4, 5], epochs=25, batch_size=32, encoder_lr=1e-5,
                    auxiliary_weights=[0.1, 0.6, 1.0], expected_cells=378,
                    precision="float32; TF32 enabled; no AMP",
                    selection="per-seed main validation BCE; auxiliary weight by ensemble validation BCE; validation Macro-F1 threshold",
                    condition="unaltered train-derived one-hot tensors from frozen paper prepared inputs",
                    auxiliary="unaltered observed targets, normalization, and heldout exclusions; uniform bucket then parent sampling")
    if getattr(args, "auxiliary_policy", None):
        from predict.baselines.repeated_runs import fixed_auxiliary_protocol
        policy = json.loads(args.auxiliary_policy.read_text())
        for cell in cells:
            fixed_auxiliary_protocol(policy, f"{cell['training_pool']}/{cell['task']}")
        manifest.update(auxiliary_policy=policy, expected_fits=len(cells) * len(MODELS) * 5 * 3,
                        selection=policy["selection"] + "; per-seed validation checkpoint and threshold")
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Run identity already binds a different experiment")
    write_json_atomic(path, manifest)
    print(json.dumps(dict(prepared_cells=len(cells), expected_results=378)))


def jobs(args, manifest):
    pending = []
    artifact_root = Path(manifest.get("artifact_root", args.output))
    policy = manifest.get("auxiliary_policy")
    for cell in manifest["cells"]:
        cell_key = f"{cell['training_pool']}/{cell['task']}"
        weights = [policy["cells"][cell_key]["weight"]] if policy else manifest["auxiliary_weights"]
        for name in args.models:
            folder = Path(name) / cell["training_pool"] / cell["task"]
            cache = artifact_root / "features" / folder
            prepared = artifact_root / cell["input"]
            feature_id = str(Path("features") / folder)
            pending.append(dict(id=feature_id, dependencies=[], marker=cache / "receipt.json",
                command=[sys.executable, "-m", "predict.baselines.foundation_suite", "features", "--model", name,
                         "--prepared", str(prepared), "--output", str(cache),
                         "--runtime-root", str(args.runtime_root), "--workers", str(args.conformer_workers if name == "unimol" else args.workers),
                         *(["--scratch-root", str(args.scratch_root.parent / "conformers")]
                           if name == "unimol" and args.scratch_root and not getattr(args, "conformer_cache", None) else []),
                         *(["--scratch-root", str(args.conformer_cache)]
                           if name == "unimol" and getattr(args, "conformer_cache", None) else [])]))
            if getattr(args, "prefetch_features", False):
                feature_job = pending[-1]
                preparation_id = str(Path("featurize") / folder)
                feature_job["dependencies"] = [preparation_id]
                command = list(feature_job["command"]) + ["--features-only"]
                command[command.index("--workers") + 1] = str(min(32, args.conformer_workers) if name == "unimol" else args.workers)
                pending.append(dict(id=preparation_id, cpu=True, preprocessing=True,
                                    dependencies=[], marker=cache / "features_ready.json", command=command))
            base = [sys.executable, "-m", "predict.baselines.minimol.run_multitask"]
            common = ["--prepared", str(prepared), "--encoder", name, "--runtime-root", str(args.runtime_root),
                      "--feature-cache", str(cache), "--workers", "0", "--threads", "2",
                      *(["--eval-batch-size", "32"] if name == "unimol" else []),
                      "--hidden-dim", str(cell["hidden_dim"]), "--depth", str(cell["depth"]),
                      "--lr", str(cell["lr"]), "--epochs", str(manifest["epochs"])]
            for arm, arm_weights in (("frozen", [0]), ("trainable", [0, *weights])):
                precision = (getattr(args, "unimol_precision", "fp32") if name == "unimol" and
                             arm == "trainable" and cell != manifest["cells"][0] else "fp32")
                precision_args = ["--precision", precision] if precision != "fp32" else []
                root = artifact_root / "runs" / folder / (arm + "_bf16" if precision == "bf16" else arm)
                dependencies = []
                for weight in arm_weights:
                    for seed in manifest["seeds"]:
                        out = root / f"aux_{weight:g}" / f"seed_{seed}"
                        job_id = str(out.relative_to(artifact_root))
                        work = args.scratch_root / job_id if args.scratch_root else out
                        dependencies.append(job_id)
                        pending.append(dict(id=job_id, dependencies=[feature_id], marker=out / "training.json", work=work,
                            exclusive=name == "unimol" and arm == "trainable" and precision == "fp32",
                            command=base + ["train", *common, "--output", str(work), "--seed", str(seed),
                                           "--aux-weight", str(weight), *precision_args, *(["--frozen"] if arm == "frozen" else [])]))
                if not policy:
                    pending.append(dict(id=str(root.relative_to(artifact_root)) + "/finalize", dependencies=dependencies,
                        marker=root / ("frozen" if arm == "frozen" else "indirect") / "metrics.json",
                        exclusive=name == "unimol" and arm == "trainable" and precision == "fp32",
                        command=base + ["finalize", *common, "--output", str(root),
                                        *precision_args, *(["--frozen"] if arm == "frozen" else [])]))
                if getattr(args, "repeated_runs", False):
                    for setting in (["frozen"] if arm == "frozen" else ["finetuned", "indirect"]):
                        for seed in manifest["seeds"]:
                            selected_weights = weights if setting == "indirect" else [0]
                            out = args.output / "repeated_runs" / folder / setting / f"seed_{seed}"
                            evaluation_id = str(out.relative_to(args.output))
                            if policy:
                                evaluation_id = evaluation_id.replace("repeated_runs/", "repeated_runs/fixed_auxiliary/", 1)
                            pending.append(dict(id=evaluation_id,
                                cpu=setting == "frozen",
                                dependencies=[str((root / f"aux_{w:g}" / f"seed_{seed}").relative_to(artifact_root))
                                              for w in selected_weights], marker=out / "metrics.json",
                                exclusive=name == "unimol" and arm == "trainable" and precision == "fp32",
                                command=[sys.executable, "-m", "predict.baselines.repeated_runs", "evaluate",
                                    "--prepared", str(prepared), "--training-root", str(root), "--encoder", name,
                                    "--runtime-root", str(args.runtime_root), "--feature-cache", str(cache),
                                    "--setting", setting, "--seed", str(seed), "--output", str(out),
                                    *(["--auxiliary-policy", str(args.output / "repeated_runs" / "auxiliary_policy.json"),
                                       "--auxiliary-cell", cell_key] if policy and setting == "indirect" else []),
                                    *(["--device", "cpu"] if setting == "frozen" else [])]))
    first = manifest["cells"][0]
    first_cohort = f"/{first['training_pool']}/{first['task']}/"
    # Exercise each new encoder early within the full queue; independent jobs
    # remain eligible while conformers or other model inputs are being prepared.
    if getattr(args, "legacy_manifest", None):
        for job in json.loads(args.legacy_manifest.read_text())["jobs"]:
            job["marker"] = Path(job["marker"])
            pending.append(job)
    if getattr(args, "prefetch_features", False):
        # Prepare inputs ahead on CPU and start long Uni-Mol fits early.
        return sorted(pending, key=lambda j: (
            0 if j["id"].startswith(("featurize/", "features/")) else
            1 if j["id"].startswith(("repeated_runs/", "legacy_repeated/")) else
            2 if j["id"].startswith("runs/unimol/") else 3,
            0 if "/unimol/" in j["id"] else 1))
    return sorted(pending, key=lambda j: (-1 if "repeated/" in j["id"] or j["id"].startswith("repeated_runs/") else
                  0 if j["id"].startswith("features/") and
                  first_cohort.rstrip("/") in j["id"] else
                  1 if first_cohort in j["id"] and j["id"].endswith("/seed_1") else 2))


def gpu_slots(job, capacity):
    if job.get("exclusive"):
        return capacity
    if "/unimol/" in job["id"] and "/frozen/" not in job["id"]:
        return (capacity + 1) // 2
    return 1


def claim_job(output, job):
    """NFS advisory lock, inherited by the worker even if its controller exits."""
    folder = output / "job_locks"
    folder.mkdir(exist_ok=True)
    handle = (folder / (hashlib.sha256(job["id"].encode()).hexdigest() + ".lock")).open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    owner = json.loads(handle.read() or "{}")
    if owner.get("state") == "complete" or owner.get("host", socket.gethostname().split(".")[0]) != socket.gethostname().split(".")[0]:
        handle.close()
        return None
    return handle


def record_claim(handle, state):
    handle.seek(0)
    handle.truncate()
    json.dump(dict(host=socket.gethostname().split(".")[0], state=state) if state else {}, handle)
    handle.flush()
    os.fsync(handle.fileno())


def run(args):
    host = socket.gethostname().split(".")[0]
    control = args.output / "controllers" / host
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((args.output / "manifest.json").read_text())
    if manifest.get("auxiliary_policy"):
        from predict.baselines.repeated_runs import fixed_auxiliary_protocol
        if not args.repeated_runs:
            raise ValueError("Fixed auxiliary runs require independent-seed evaluation")
        policy = manifest["auxiliary_policy"]
        for cell in manifest["cells"]:
            key = f"{cell['training_pool']}/{cell['task']}"
            fixed_auxiliary_protocol(policy, key)
            for source in policy["cells"][key]["sources"]:
                if sha256_file(Path(source["path"])) != source["sha256"]:
                    raise ValueError("MiniMol auxiliary selection source changed")
        path = args.output / "repeated_runs" / "auxiliary_policy.json"
        if path.exists() and json.loads(path.read_text()) != policy:
            raise ValueError("Fixed auxiliary policy changed within a run")
        write_json_atomic(path, policy)
    if args.repeated_runs:
        from predict.baselines.repeated_runs import PROTOCOL
        write_json_atomic(args.output / "repeated_runs" / "protocol.json", dict(PROTOCOL,
            auxiliary_lambda=manifest["auxiliary_policy"]["selection"]) if manifest.get("auxiliary_policy") else PROTOCOL)
    execution = dict(unimol_precision=args.unimol_precision,
        fp32_retained_cohort={k: manifest["cells"][0][k] for k in ("training_pool", "task")},
        policy="Existing first-cohort FP32 fits retained; new Uni-Mol trainable cohorts use distinct precision-bound directories")
    policy_path = args.output / "execution_policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != execution:
        raise ValueError("Execution precision policy changed; preserve its original run identity")
    write_json_atomic(policy_path, execution)
    if json.loads((args.runtime_root / "manifest.json").read_text())["models"] != manifest["models"]:
        raise ValueError("Runtime model releases changed")
    all_jobs = jobs(args, manifest)
    jobs_by_id = {j["id"]: j for j in all_jobs}
    for job in all_jobs:
        publish_fit(job)
    done = {job["id"] for job in all_jobs if job["marker"].exists()}
    pending = [job for job in all_jobs if job["id"] not in done]
    active, failures = [], []
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    # A controller can die while its workers finish/resume independently.
    # Adopt only exact commands from this run, preventing duplicate GPU fits.
    commands = {tuple(j["command"][1:]): j for j in pending}
    for process in Path("/proc").glob("[0-9]*"):
        try:
            argv = process.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
            job = commands.get(tuple(argv[1:]))
            if job is None:
                continue
            parent = process.joinpath("stat").read_text().split(")", 1)[1].split()[1]
            parent_command = Path(f"/proc/{parent}/cmdline")
            if parent_command.exists() and parent_command.read_bytes().decode().strip("\0").split("\0") == argv:
                continue  # A preprocessing pool worker, not its controlling process.
            gpu = None if job.get("cpu") else int(next(v.split("=", 1)[1]
                for v in process.joinpath("environ").read_bytes().decode().split("\0") if v.startswith("CUDA_VISIBLE_DEVICES=")))
        except (OSError, ValueError, StopIteration):
            continue
        pending.remove(job)
        del commands[tuple(argv[1:])]
        path = logs / (job["id"].replace("/", "__") + ".log")
        task_lock = claim_job(args.output, job)
        if task_lock is not None:
            record_claim(task_lock, "running")
        else:
            # This exact surviving local worker owns the inherited lock. Record
            # its host so SSD publication remains reserved after the worker exits.
            claim_path = args.output / "job_locks" / (hashlib.sha256(job["id"].encode()).hexdigest()+".lock")
            with claim_path.open("r+") as inherited:
                record_claim(inherited, "running")
        active.append(dict(process=ExistingProcess(int(process.name), argv), log=path.open("a"),
                           job=job, gpu=gpu, path=path, lock=task_lock))
    last_report = 0
    published = 0
    last_audit = 0
    audit = None
    publishing = []
    archives = ThreadPoolExecutor(max_workers=2)
    reports = ThreadPoolExecutor(max_workers=1)
    write_json_atomic(control / "execution_resources.json", dict(
        slots_per_gpu=args.slots_per_gpu, unimol_slots=(args.slots_per_gpu + 1) // 2,
        cpu_slots=args.cpu_slots, feature_cpu_slots=args.feature_cpu_slots,
        prefetch_features=args.prefetch_features, local_cache=os.environ.get("TXAGENT_FOUNDATION_CACHE"),
        archive_workers=2, audit_interval=args.audit_interval))

    def event(kind, entry):
        with (control / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(time=time.time(), event=kind, job=entry["job"]["id"],
                                         gpu=entry["gpu"], pid=entry["process"].pid)) + "\n")

    def complete(entry, code):
        job = entry["job"]
        completed = job["marker"].exists()
        task_lock = entry.get("lock") or claim_job(args.output, job)
        if task_lock:
            record_claim(task_lock, "complete" if completed else None)
            task_lock.close()
        if completed:
            done.add(job["id"])
            event("published", entry)
        else:
            failures.append(dict(job=job["id"], exit_code=None if isinstance(entry["process"], ExistingProcess) else code,
                                 log=str(entry["path"])))

    def launch(eligible, gpu, task_lock):
        pending.remove(eligible)
        path = logs / (eligible["id"].replace("/", "__") + ".log")
        handle = path.open("a")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu), OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        env.update(eligible.get("env", {}))
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        record_claim(task_lock, "running")
        proc = subprocess.Popen(eligible["command"], cwd=args.output, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, pass_fds=(task_lock.fileno(),))
        active.append(dict(process=proc, log=handle, job=eligible, gpu=gpu, path=path, lock=task_lock))
        event("started", active[-1])

    while pending or active or publishing:
        for entry, future in list(publishing):
            if future.done():
                future.result()
                publishing.remove((entry, future))
                complete(entry, 0)
        if audit is not None and audit.done():
            if audit.result():
                published = auditing_count
            audit = None
        for entry in list(active):
            code = entry["process"].poll()
            if code is None:
                continue
            entry["log"].close()
            active.remove(entry)
            job = entry["job"]
            event("worker_finished", entry)
            if code == 0 and job.get("work"):
                publishing.append((entry, archives.submit(publish_fit, job)))
            else:
                complete(entry, code)
        # Completion receipts are shared across hosts; scratch checkpoints are local.
        done.update(j["id"] for j in pending if j["marker"].exists())
        # Externally recovered evaluations count toward completion as well.
        done.update(f["job"] for f in failures if jobs_by_id[f["job"]]["marker"].exists())
        failures = [f for f in failures if f["job"] not in done]
        pending = [j for j in pending if j["id"] not in done]
        failed_ids = {f["job"] for f in failures}
        for job in list(pending):
            if any(d in failed_ids for d in job["dependencies"]):
                pending.remove(job)
                failed_ids.add(job["id"])
                failures.append(dict(job=job["id"], exit_code=None, reason="failed dependency"))
        ready, claims = [], {}
        free_slots = sum(max(0, args.slots_per_gpu - sum(gpu_slots(a["job"], args.slots_per_gpu)
                         for a in active if a["gpu"] == gpu)) for gpu in args.gpus)
        if len(publishing) >= 8:
            free_slots = 0  # Bound unpublished SSD work if shared storage falls behind.
        limits = {False: free_slots * 2, True: max(0, args.cpu_slots - sum(a["gpu"] is None for a in active)) * 2}
        claimed = {False: 0, True: 0}
        for job in pending:
            if all(claimed[k] >= limits[k] for k in limits):
                break
            cpu = bool(job.get("cpu"))
            if job.get("preprocessing") and sum(a["job"].get("preprocessing", False) for a in active) + sum(j.get("preprocessing", False) for j in ready) >= args.feature_cpu_slots:
                continue
            if claimed[cpu] >= limits[cpu]:
                continue
            if all(d in done for d in job["dependencies"]):
                handle = claim_job(args.output, job)
                if handle is not None:
                    if job["marker"].exists():
                        handle.close()
                    else:
                        ready.append(job)
                        claims[job["id"]] = handle
                        claimed[cpu] += 1
        for job in list(ready):
            if job.get("cpu") and sum(a["gpu"] is None for a in active) < args.cpu_slots:
                ready.remove(job)
                launch(job, None, claims.pop(job["id"]))
        large = sum(j.get("exclusive", False) for j in ready)
        available = [g for g in args.gpus if not any(a["gpu"] == g and a["job"].get("exclusive") for a in active)]
        reserved = set(sorted(available, key=lambda g: sum(a["gpu"] == g for a in active))[:large])
        for gpu in args.gpus:
            while True:
                used = sum(gpu_slots(a["job"], args.slots_per_gpu) for a in active if a["gpu"] == gpu)
                if used >= args.slots_per_gpu or (gpu in reserved and used):
                    break
                eligible = next((j for j in ready if not j.get("cpu") and
                    used + gpu_slots(j, args.slots_per_gpu) <= args.slots_per_gpu and
                    (j.get("exclusive") if gpu in reserved else not j.get("exclusive") or used == 0)), None)
                if eligible is None:
                    break
                ready.remove(eligible)
                launch(eligible, gpu, claims.pop(eligible["id"]))
        for handle in claims.values():
            handle.close()
        if time.time() - last_report >= 30 or not active:
            status = dict(host=host, status="running" if active or pending or publishing else "failed" if failures else "complete",
                          completed_jobs=len(done), total_jobs=len(all_jobs), queued=len(pending),
                          publishing=[e["job"]["id"] for e, _ in publishing], audit_running=audit is not None,
                          active=[dict(job=a["job"]["id"], pid=a["process"].pid, gpu=a["gpu"]) for a in active],
                          failures=failures, updated_unix=time.time())
            status["completed_fits"] = sum(j["marker"].name == "training.json" and j["id"] in done for j in all_jobs)
            write_json_atomic(control / "status.json", status)
            peers = [json.loads(p.read_text()) for p in (args.output / "controllers").glob("*/status.json")]
            write_json_atomic(args.output / "status.json", dict(status, controllers=peers,
                active=[dict(a, host=p["host"]) for p in peers for a in p["active"]],
                failures=[dict(f, host=p["host"]) for p in peers for f in p["failures"]]))
            with (control / "resources.jsonl").open("a") as handle:
                gpu_state = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                handle.write(json.dumps(dict(time=time.time(), gpus=gpu_state.stdout, active=status["active"])) + "\n")
            print(json.dumps({k: v for k, v in status.items() if k != "active"}), flush=True)
            last_report = time.time()
            finished_evaluations = sum(j.endswith("/finalize") or j.startswith(("repeated_runs/", "legacy_repeated/")) for j in done)
            if finished_evaluations > published and audit is None and time.time() - last_audit >= args.audit_interval:
                auditing_count = finished_evaluations
                audit = reports.submit(audit_reports, args, blocking=False)
                last_audit = time.time()
        if not active and not pending and not publishing:
            break
        time.sleep(2)
    archives.shutdown(wait=True)
    reports.shutdown(wait=True)
    if audit is not None:
        audit.result()
    audit_reports(args)
    if failures or pending:
        raise RuntimeError("Some jobs failed or have unmet dependencies; inspect status.json and resume after repair")


def audit_reports(args, *, blocking=True):
    with (args.output / "verify.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            return False
        manifest = json.loads((args.output / "manifest.json").read_text())
        if not manifest.get("auxiliary_policy"):
            verify(args)
        if args.repeated_runs:
            from predict.baselines.repeated_runs import summarize
            summarize(args.output / "repeated_runs", 378)
            if args.legacy_manifest:
                legacy = json.loads(args.legacy_manifest.read_text())
                summarize(Path(legacy["output"]), 54)
                summarize(args.output / "repeated_runs" / "combined", 432,
                          sources=[args.output / "repeated_runs", Path(legacy["output"])])
    return True


class ExistingProcess:
    def __init__(self, pid, argv):
        self.pid, self.argv = pid, argv

    def poll(self):
        try:
            argv = Path(f"/proc/{self.pid}/cmdline").read_bytes().decode().strip("\0").split("\0")
            return None if argv == self.argv else 0
        except FileNotFoundError:
            return 0


def publish_fit(job):
    """Recover finished SSD work after interruption; publish the receipt last."""
    work = job.get("work")
    if not work or work == job["marker"].parent or not (work / "training.json").exists():
        return
    destination = job["marker"].parent
    if job["marker"].exists():
        if json.loads(job["marker"].read_text()) != json.loads((work / "training.json").read_text()):
            raise ValueError(f"Conflicting completed SSD/archive fits: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(work.iterdir(), key=lambda p: p.name == "training.json"):
            temporary = destination / (source.name + ".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(destination / source.name)
        for name in ("resume.pt", "resume.tmp"):
            (destination / name).unlink(missing_ok=True)
    shutil.rmtree(work)


def verify(args):
    manifest = json.loads((args.output / "manifest.json").read_text())
    if manifest.get("auxiliary_policy"):
        from predict.baselines.repeated_runs import summarize
        summarize(args.output / "repeated_runs", 378)
        return
    results, prepared_data = [], {}
    for name in manifest["models"]:
        for cell in manifest["cells"]:
            prepared = args.output / cell["input"]
            for setting in manifest["settings"]:
                root = args.output / "runs" / name / cell["training_pool"] / cell["task"]
                accelerated = root / "trainable_bf16" / setting
                path = (accelerated if setting != "frozen" and accelerated.exists() else
                        root / ("frozen" if setting == "frozen" else "trainable") / setting)
                if not (path / "metrics.json").exists():
                    continue
                if prepared not in prepared_data:
                    prepared_data[prepared] = torch.load(prepared / "data.pt", map_location="cpu", weights_only=False)
                data = prepared_data[prepared]
                metrics = json.loads((path / "metrics.json").read_text())
                selection = metrics["selection"]
                if selection["test_used"] or selection["seeds"] != manifest["seeds"]:
                    raise ValueError(f"Wrong selection protocol: {path}")
                weights = [0] if setting == "frozen" else [0, *manifest["auxiliary_weights"]]
                candidate_scores = {}
                for weight in weights:
                    members = []
                    for seed in manifest["seeds"]:
                        member = path.parent / f"aux_{weight:g}" / f"seed_{seed}"
                        receipt = json.loads((member / "training.json").read_text())
                        best = min(receipt["history"], key=lambda r: r["valid_bce"])
                        if (receipt["test_used"] or len(receipt["history"]) != manifest["epochs"]
                                or best["epoch"] != receipt["best_epoch"]):
                            raise ValueError(f"Invalid checkpoint selection: {member}")
                        scores = np.load(member / "valid_scores.npy")
                        if not np.isclose(ensemble_bce(data["splits"]["valid"]["labels"].numpy(), scores),
                                          best["valid_bce"], rtol=1e-4, atol=1e-5):
                            raise ValueError(f"Saved validation scores disagree with selected epoch: {member}")
                        members.append(scores)
                    candidate_scores[weight] = np.mean(members, axis=0)
                if setting != "frozen":
                    losses = {w: ensemble_bce(data["splits"]["valid"]["labels"].numpy(), candidate_scores[w])
                              for w in manifest["auxiliary_weights"]}
                    chosen = min(losses, key=lambda w: (losses[w], w))
                    if chosen != selection["selected_aux_weight"] or any(
                            not np.isclose(v, selection["candidates"][str(w)]) for w, v in losses.items()):
                        raise ValueError(f"Auxiliary weight was not selected on validation: {path}")
                expected_weight = selection["selected_aux_weight"] if setting == "indirect" else 0
                if metrics["aux_weight"] != expected_weight:
                    raise ValueError(f"Wrong selected setting: {path}")
                for split in ("valid", "test"):
                    file = path / f"{split}_predictions.jsonl"
                    if sha256_file(file) != metrics["prediction_sha256"][split]:
                        raise ValueError(f"Prediction bytes changed: {file}")
                    rows = [json.loads(line) for line in file.read_text().splitlines()]
                    expected = data["splits"][split]
                    if len(rows) != len(expected["indices"]) or [r["index"] for r in rows] != list(range(len(rows))):
                        raise ValueError(f"Missing/duplicate predictions: {file}")
                    for i, row in enumerate(rows):
                        if row["drug"] != data["smiles"][int(expected["indices"][i])] or row["label"] != int(expected["labels"][i]):
                            raise ValueError(f"Prediction identity mismatch: {file}")
                        if row["prediction"] != int(row["probability"] >= metrics["threshold"]):
                            raise ValueError(f"Incorrect threshold application: {file}")
                        if "condition_groups" in expected and row["condition_group"] != expected["condition_groups"][i]:
                            raise ValueError(f"Changed condition identity: {file}")
                    scores = np.asarray([r["probability"] for r in rows])
                    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                        raise ValueError(f"Invalid probabilities: {file}")
                    labels = expected["labels"].tolist()
                    if split == "valid":
                        if not np.allclose(scores, candidate_scores[expected_weight], rtol=0, atol=1e-7):
                            raise ValueError(f"Validation prediction is not the selected five-seed ensemble: {file}")
                        threshold, _ = choose_threshold(labels, scores, "valid_macro_f1")
                        if threshold != metrics["threshold"]:
                            raise ValueError("Threshold was not selected on validation")
                    actual = evaluate_metrics(labels, scores, metrics["threshold"])
                    if any(not np.isclose(v, metrics[split][k], equal_nan=True) for k, v in actual.items()):
                        raise ValueError(f"Metric mismatch: {file}")
                results.append(dict(model=name, training_pool=cell["training_pool"], benchmark=cell["benchmark"],
                                    task=cell["task"], setting=setting, **metrics["test"],
                                    auxiliary_weight=metrics["aux_weight"],
                                    predictions=str((path / "test_predictions.jsonl").relative_to(args.output))))
    write_json_atomic(args.output / "results.json", results)
    if results:
        with atomic_output_path(args.output / "scores.csv") as temporary:
            with temporary.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
    write_json_atomic(args.output / "verification.json", dict(verified_cells=len(results), expected_cells=378,
                      complete=len(results) == 378, scope="checkpoint/lambda validation selection, five-seed ensembles, molecule/condition identity, thresholds and recomputed metrics"))
    print(json.dumps(dict(verified_cells=len(results), expected_cells=378)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "features", "conformers", "run", "verify"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path)
    p.add_argument("--sources", type=Path, default=Path("experiments/baselines/foundation_sources.json"))
    p.add_argument("--source-root", type=Path)
    p.add_argument("--run-name")
    p.add_argument("--auxiliary-policy", type=Path, help="MiniMol-derived fixed-lambda policy for a fresh prepared run")
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--prepared", type=Path)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--conformer-workers", type=int, default=64)
    p.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--slots-per-gpu", type=int, default=2)
    p.add_argument("--scratch-root", type=Path, help="Local SSD checkpoint staging; completed fits are published to --output")
    p.add_argument("--repeated-runs", action="store_true")
    p.add_argument("--legacy-manifest", type=Path)
    p.add_argument("--unimol-precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--conformer-cache", type=Path)
    p.add_argument("--partition", type=int, default=0)
    p.add_argument("--partitions", type=int, default=1)
    p.add_argument("--cpu-slots", type=int, default=8)
    p.add_argument("--prefetch-features", action="store_true", help="Build molecular inputs in CPU jobs ahead of GPU embedding/training")
    p.add_argument("--features-only", action="store_true", help="Prepare molecular inputs without GPU embedding inference")
    p.add_argument("--feature-cpu-slots", type=int, default=2)
    p.add_argument("--audit-interval", type=float, default=120, help="Seconds between background result audits")
    args = p.parse_args()
    args.output = args.output.resolve()
    for name in ("runtime_root", "source_root", "sources", "prepared", "scratch_root", "legacy_manifest", "conformer_cache"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    required = {"prepare": ("runtime_root", "source_root", "run_name"),
                "features": ("runtime_root", "prepared", "model"), "conformers": ("runtime_root", "conformer_cache"),
                "run": ("runtime_root",), "verify": ()}
    if any(getattr(args, name) is None for name in required[args.phase]):
        p.error(f"{args.phase} requires {', '.join(required[args.phase])}")
    if len(set(args.models)) != len(args.models) or len(set(args.gpus)) != len(args.gpus):
        p.error("models and GPUs must be distinct")
    if min(args.workers, args.conformer_workers, args.slots_per_gpu, args.cpu_slots, args.feature_cpu_slots, args.audit_interval) <= 0 or min(args.gpus) < 0:
        p.error("workers/slots must be positive and GPU indices nonnegative")
    torch.set_num_threads(2)
    if args.phase == "conformers":
        from predict.baselines.foundation_models import warm_conformers
        if not 0 <= args.partition < args.partitions:
            p.error("Invalid conformer partition")
        torch.set_num_threads(1)
        warm_conformers(args.runtime_root, args.output, args.conformer_cache, args.conformer_workers, args.partition, args.partitions)
    elif args.phase == "features":
        prepare_features(args.model, args.runtime_root, args.prepared, args.output, args.workers,
                         batch_size=32 if args.model == "unimol" else 128, scratch=args.scratch_root,
                         features_only=args.features_only)
    else:
        {"prepare": prepare, "run": run, "verify": verify}[args.phase](args)


if __name__ == "__main__":
    main()
