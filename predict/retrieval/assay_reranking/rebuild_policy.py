"""Prepare, score and publish immutable successors for a named molecular policy."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.paths import REPO_ROOT, resolve_legacy_path
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic
from predict.retrieval.policies import MOLECULAR_REPRESENTATION_POLICY
from . import build_ranked_uid_retrieval as builder
from .artifact_bundle import build_bundle_manifest, verify_bundle


MODULE = "predict.retrieval.assay_reranking.build_ranked_uid_retrieval"


def inventory(study: Path, catalogs: list[Path]) -> dict:
    if (study / "study.json").exists():
        raise FileExistsError(study / "study.json")
    sources = [json.loads(path.read_text()) for path in catalogs]
    jobs = []
    for benchmark, tasks in sources[1]["benchmarks"].items():
        for task, entry in tasks.items():
            release = "v10_main_universe_v5" if task == "skin_reaction" else "v10_main_universe_v3"
            projection = Path("data/evidence_libraries") / task / release / "retrieval_projection/ranked_evidence_source_v1/VERSION.json"
            direct_index = REPO_ROOT / entry["retrieval"]["L1"]
            direct = json.loads(direct_index.read_text())
            old_direct_manifest = (direct_index.parent / direct["splits"]["test"]["levels"]["L1"]["manifest"]).resolve()
            gold = REPO_ROOT / "data/gold_labels" / builder.GOLD_NAMES[task] / "v1/scaffold"
            train_root = gold if benchmark == "gold_v1" else builder.tdc_task_root(task)
            l1_output = Path(f"data/caches/assay_reranking/active/flat_v5/{benchmark}/{task}/l1/assay_transfer/frozen_context/source_v3")
            direct_evidence = (direct_index.parent / direct["evidence"]["manifest"]).resolve()
            evidence = json.loads(direct_evidence.read_text())
            coverage = []
            for subset in ("test", "valid"):
                path = builder._score_manifest(direct_index, task, subset, "L1")
                manifest = json.loads(path.read_text())
                with sqlite3.connect(f"file:{path.with_name(manifest['database'])}?mode=ro&immutable=1", uri=True) as connection:
                    coverage.append(connection.execute("SELECT COUNT(assay_transfer_score),COUNT(*) FROM rankings").fetchone())
            has_model = json.loads(old_direct_manifest.read_text()).get("model") is not None
            jobs.append(dict(id=f"{benchmark}_{task}_l1", task=task, benchmark=benchmark, role="l1",
                             policy="scaffold_disjoint", levels=["L1"], release=release,
                             output=str(l1_output), evidence=str(l1_output / task / "evidence/VERSION.json"),
                             selection_output=f"data/caches/assay_reranking/active/flat_v5/{benchmark}/{task}/l1_cards/frozen_context_source_v1",
                             train_root=str(train_root), membership=str(gold / "voter_membership.parquet") if benchmark == "gold_v1" else None,
                             membership_projection=str(direct_evidence.with_name(evidence["records"])),
                             direct_model_manifest=str(old_direct_manifest),
                             morgan_only=has_model and not any(scored for scored, total in coverage),
                             preserve_uncomputed_queries=has_model and any(scored < total for scored, total in coverage),
                             reuse=[str(direct_index)] if has_model else []))
            for n, policy in enumerate(("parent_disjoint", "scaffold_disjoint")):
                old_entry = sources[n]["benchmarks"][benchmark][task]
                old_path = REPO_ROOT / old_entry["retrieval"]["later"]
                old = json.loads(old_path.read_text())
                old_level_paths = [(old_path.parent / item["manifest"]).resolve()
                                   for item in old["splits"]["test"]["levels"].values()
                                   if Path(item["manifest"]).parent.name != "L1"]
                reuse = set()
                for candidate in sources:
                    source = REPO_ROOT / candidate["benchmarks"][benchmark][task]["retrieval"]["later"]
                    index = json.loads(source.read_text())
                    for item in index["splits"]["test"]["levels"].values():
                        path = resolve_legacy_path((source.parent / item["manifest"]).resolve())
                        if path.parent.name != "L1" and json.loads(path.read_text()).get("model"):
                            reuse.add(str(source))
                version = "v2" if policy == "parent_disjoint" else "v3"
                model_release = {"bbb_martins": "v24_1", "bioavailability_ma": "v25"}.get(task, "v27")
                profile = f"flat_v5/{benchmark}/{task}/l2plus/assay_transfer/{model_release}/{policy}_parent100_{version}"
                jobs.append(dict(id=f"{benchmark}_{task}_{policy}", task=task, benchmark=benchmark,
                                 role="l2plus", policy=policy, levels=[x for x in old["splits"]["test"]["levels"] if x != "L1"],
                                 release=release, output=str(Path("data/caches/assay_reranking/active") / profile),
                                 evidence=str(projection), profile=profile,
                                 morgan_only=all(json.loads(path.read_text()).get("model") is None for path in old_level_paths),
                                 query_identity_root=str(old_level_paths[0].parents[4]), reuse=sorted(reuse)))
    for job in jobs:
        if job["role"] != "l2plus":
            continue
        benchmark, task = job["benchmark"], job["task"]
        reference = REPO_ROOT / sources[1]["benchmarks"][benchmark][task]["retrieval"]["later"]
        native = REPO_ROOT / "data/gold_labels" / builder.GOLD_NAMES[task] / "v1/scaffold" if benchmark == "gold_v1" else builder.tdc_task_root(task)
        job["query_identity_root"] = None
        job["query_identity_source"] = "native frozen benchmark metadata"
        job["predecessor_query_cohort_matches"] = all(
            json.loads(builder._score_manifest(reference, task, split, job["levels"][0]).read_text())["inputs"]["query_sha256"]
            == sha256_file(native / f"{split}_molecule_condition_labels.jsonl") for split in ("valid", "test"))
        job["reuse_by_level"] = {}
        for level in job["levels"]:
            expected = json.loads(builder._score_manifest(reference, task, "test", level).read_text())["model"]
            roots = []
            if not job["morgan_only"] and expected is not None:
                benchmarks = [benchmark, *(name for name in sources[1]["benchmarks"] if name != benchmark)]
                for source_benchmark in benchmarks:
                    for catalog in reversed(sources):
                        index = REPO_ROOT / catalog["benchmarks"][source_benchmark][task]["retrieval"]["later"]
                        manifest = builder._score_manifest(index, task, "test", level)
                        if manifest.is_file() and json.loads(manifest.read_text()).get("model") == expected:
                            if str(index.resolve()) not in roots:
                                roots.append(str(index.resolve()))
            job["reuse_by_level"][level] = roots
    value = dict(status="building", molecular_representation_policy=MOLECULAR_REPRESENTATION_POLICY,
                 source_catalogs={str(path): sha256_file(path) for path in catalogs}, jobs=jobs)
    write_json_atomic(study / "study.json", value)
    return value


def invoke(study: Path, job: dict, action: str, subset: str | None = None,
           level: str | None = None, extra: list[str] = ()) -> None:
    command = [sys.executable, "-m", MODULE, action, "--task", job["task"],
               "--benchmark", "gold" if job["benchmark"] == "gold_v1" else "tdc",
               "--neighbor-identity-policy", job["policy"], "--evidence-release", job["release"],
               "--molecular-representation-policy", MOLECULAR_REPRESENTATION_POLICY,
               "--profile", job["profile"] if "profile" in job else str(Path(job["output"]).relative_to("data/caches/assay_reranking/active")),
               "--output-root", job["output"], "--evidence-manifest", job["evidence"]]
    if subset:
        command += ["--subset", subset]
    if level or job["role"] == "l1":
        command += ["--level", level or "L1"]
    if job["role"] == "l1":
        command += ["--direct-model-manifest", job["direct_model_manifest"]]
        if job.get("morgan_only"):
            command += ["--morgan-only"]
        elif job.get("preserve_uncomputed_queries"):
            command += ["--preserve-uncomputed-queries"]
        if action == "prepare-level":
            for root in job.get("reuse_by_level", {}).get(level, job["reuse"]):
                command += ["--score-reuse-root", root]
    else:
        if job.get("query_identity_root"):
            command += ["--query-identity-root", job["query_identity_root"]]
        if job["morgan_only"]:
            command += ["--morgan-only"]
        if action in {"prepare-level", "reuse-scores"} and not job["morgan_only"]:
            command += ["--prefer-first-score-source"]
            roots = list(job.get("reuse_by_level", {}).get(level, job["reuse"]))
            expected = (json.loads(builder._score_manifest(Path(roots[0]), job["task"], "test", level).read_text())["model"]
                        if roots else None)
            for candidate in json.loads((study / "study.json").read_text())["jobs"]:
                if expected is None or candidate["task"] != job["task"] or candidate["role"] != "l2plus":
                    continue
                index = REPO_ROOT / candidate["output"] / job["task"] / "RELEASE_INDEX.json"
                for split in ("test", "valid"):
                    path = builder._score_manifest(index, job["task"], split, level)
                    if not path.is_file():
                        continue
                    manifest = json.loads(path.read_text())
                    if manifest.get("status") == "complete" and manifest.get("model") == expected and str(index) not in roots:
                        roots.append(str(index))
            for root in roots:
                command += ["--score-reuse-root", root]
    command += list(extra)
    shard = extra[extra.index("--shard-index") + 1] if "--shard-index" in extra else "all"
    path = study / "logs" / job["id"] / f"{subset or 'release'}_{level or 'all'}_{action}_{shard}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if action == "score":
        environment["HF_MODULES_CACHE"] = str((study / ".runtime" / "hf_modules" / job["id"] / path.stem).resolve())
    with path.open("a") as log:
        result = subprocess.run(command, cwd=REPO_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{job['id']}: {action} failed; see {path}")


def prepare(study: Path, job: dict) -> None:
    output = REPO_ROOT / job["output"]
    if job["role"] == "l1" and not (output / job["task"] / "RELEASE_INDEX.json").exists():
        from optimization.selection.pool import prepare_l1_cache
        train = Path(job["train_root"])
        prepare_l1_cache(job["task"], train / "train_molecule_condition_labels.jsonl",
                         {s: train / f"{s}_molecule_condition_labels.jsonl" for s in ("test", "valid")},
                         output / job["task"], membership_path=Path(job["membership"]) if job["membership"] else None,
                         membership_projection=Path(job["membership_projection"]),
                         molecular_representation_policy=MOLECULAR_REPRESENTATION_POLICY,
                         include_chirality=False)
    if job["role"] == "l1" and not (REPO_ROOT / job["selection_output"] / "RELEASE_INDEX.json").exists():
        from optimization.selection.pool import prepare_l1_cache
        train = Path(job["train_root"])
        prepare_l1_cache(job["task"], train / "train_molecule_condition_labels.jsonl",
                         {s: train / f"{s}_molecule_condition_labels.jsonl" for s in ("test", "valid")},
                         REPO_ROOT / job["selection_output"],
                         membership_path=Path(job["membership"]) if job["membership"] else None,
                         membership_projection=Path(job["membership_projection"]),
                         molecular_representation_policy=MOLECULAR_REPRESENTATION_POLICY,
                         include_chirality=True)
    for subset in ("test", "valid"):
        if job["role"] == "l2plus" and job["task"] in {"skin_reaction", *builder.ADDON_TASKS}:
            universe = output / job["task"] / "scaffold" / subset / "PARENT_UNIVERSE.json"
            if not universe.exists():
                invoke(study, job, "prepare-universe", subset)
        for level in job["levels"]:
            path = output / job["task"] / "scaffold" / subset / level / "VERSION.json"
            if job["role"] == "l1" and job.get("morgan_only"):
                continue
            if not path.exists() or (job["role"] == "l1" and json.loads(path.read_text()).get("model") is None):
                invoke(study, job, "prepare-level", subset, level)
            manifest = json.loads(path.read_text())
            if job["role"] == "l2plus" and manifest["status"] == "prepared" and manifest.get("score_reuse_policy") != "first_declared_source":
                invoke(study, job, "reuse-scores", subset, level)
                manifest = json.loads(path.read_text())
            if manifest["status"] == "ready_to_finalize":
                invoke(study, job, "finalize", subset, level)
    invoke(study, job, "index")


def prepare_score_worksets(study: Path, jobs: list[dict], output: Path) -> dict:
    """Deduplicate compatible pending prompts across cache policies and cohorts."""
    jobs = [j for j in jobs if j["role"] == "l2plus"]
    if not jobs or len({j["task"] for j in jobs}) != 1 or output.exists():
        raise ValueError("Score worksets require one task and a fresh output root")
    levels = {}
    sources = []
    for job in jobs:
        for subset in ("test", "valid"):
            for level in job["levels"]:
                path = REPO_ROOT / job["output"] / job["task"] / "scaffold" / subset / level / "VERSION.json"
                manifest = json.loads(path.read_text())
                if manifest["status"] in {"complete", "ready_to_finalize"}:
                    continue
                if (manifest["status"] != "prepared" or manifest.get("score_reuse_policy") != "first_declared_source"
                        or list((path.parent / ".scores").glob("*.jsonl"))):
                    raise ValueError(f"Finish existing scoring and reconcile reusable scores first: {path}")
                group = levels.setdefault(level, dict(model=manifest["model"],
                    contract=manifest["scoring_contract_version"], prompts={}))
                if (group["model"], group["contract"]) != (manifest["model"], manifest["scoring_contract_version"]):
                    raise ValueError(f"Incompatible scoring models or contracts at {level}")
                prompt_path = path.parent / manifest["prompts"]
                for row in pq.read_table(prompt_path).to_pylist():
                    key = row["score_key"]
                    if key in group["prompts"] and group["prompts"][key] != row:
                        raise ValueError(f"One score key has different rendered prompts: {key}")
                    group["prompts"][key] = row
                sources.append(dict(manifest=str(path), manifest_sha256=sha256_file(path),
                                    prompts_sha256=sha256_file(prompt_path)))
    task = jobs[0]["task"]
    index = dict(schema_version="assay_transfer_score_workset_index.v1", task_id=task,
                 splits={"test": {"levels": {}}})
    for level, group in levels.items():
        target = output / task / "scaffold/test" / level
        target.mkdir(parents=True)
        rows = [group["prompts"][key] for key in sorted(group["prompts"])]
        pq.write_table(pa.Table.from_pylist(rows), target / "prompts.parquet", compression="zstd")
        write_json_atomic(target / "VERSION.json", dict(schema_version="assay_transfer_score_workset.v1",
            status="prepared", task_id=task, level=level, model=group["model"],
            scoring_contract_version=group["contract"], prompts="prompts.parquet", score_key_count=len(rows)))
        index["splits"]["test"]["levels"][level] = dict(manifest=f"scaffold/test/{level}/VERSION.json")
    write_json_atomic(output / task / "RELEASE_INDEX.json", index)
    value = dict(source_study=str(study), source_preparations=sources, jobs=[
        {**jobs[0], "output": str(output), "morgan_only": False, "subsets": ["test"],
         "levels": sorted(levels), "profile": "molecular_policy_score_workset"}])
    write_json_atomic(output / "study.json", value)
    return value


def publish(study: Path, build: dict) -> None:
    for job in build["jobs"]:
        invoke(study, job, "index")
        invoke(study, job, "validate")
    direct = {(j["benchmark"], j["task"]): j for j in build["jobs"] if j["role"] == "l1"}
    for policy, version in (("parent_disjoint", "v2"), ("scaffold_disjoint", "v3")):
        catalog = dict(schema_version="retrieval_cache_set.v1", name=f"{policy}_parent100", release=version,
                       molecular_representation_policy=MOLECULAR_REPRESENTATION_POLICY,
                       indirect_identity_policy=policy, benchmarks={})
        for job in build["jobs"]:
            if job["role"] != "l2plus" or job["policy"] != policy:
                continue
            l1 = direct[job["benchmark"], job["task"]]
            index = str(Path(l1["output"]) / job["task"] / "RELEASE_INDEX.json")
            catalog["benchmarks"].setdefault(job["benchmark"], {})[job["task"]] = dict(
                retrieval={"L1": index, "later": str(Path(job["output"]) / job["task"] / "RELEASE_INDEX.json")},
                selection=dict(direct_cache=str(Path(l1["selection_output"]) / "RELEASE_INDEX.json"),
                               context_identity="canonical_context_molecule.v1", direct_assay_scores="not_computed"))
        root = REPO_ROOT / "predict/retrieval/assay_reranking/flat_v5_manifests" / f"{policy}_parent100_{version}"
        path = root / "catalog.json"
        if path.exists():
            raise FileExistsError(path)
        write_json_atomic(path, catalog)
        for benchmark, tasks in catalog["benchmarks"].items():
            for task in tasks:
                bundle = root / benchmark / f"{task}.json"
                build_bundle_manifest(task=task, benchmark=benchmark, cache_config=path, output=bundle)
                verify_bundle(bundle)
    build["status"] = "complete"
    write_json_atomic(study / "study.json", build)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("inventory", "prepare", "reuse", "worksets", "tokenize", "score", "finalize", "publish"))
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--source-catalog", type=Path, action="append")
    parser.add_argument("--job", action="append")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard", type=int, action="append")
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--score-reuse-root", type=Path, action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.6)
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if args.phase == "inventory":
        if not args.source_catalog or len(args.source_catalog) != 2:
            parser.error("Provide parent then scaffold source catalogs")
        inventory(args.study, args.source_catalog)
        return
    build = json.loads((args.study / "study.json").read_text())
    jobs = [j for j in build["jobs"] if not args.job or j["id"] in args.job]
    if args.job and len(jobs) != len(set(args.job)):
        parser.error("Unknown or repeated job ID")
    for job in jobs:
        if job["role"] == "l2plus" and args.score_reuse_root:
            job["reuse_by_level"] = {level: [*job.get("reuse_by_level", {}).get(level, job["reuse"]),
                *(str(path) for path in args.score_reuse_root)] for level in job["levels"]}
    if args.phase == "prepare":
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda job: prepare(args.study, job), jobs))
    elif args.phase == "worksets":
        if args.output_root is None:
            parser.error("Worksets require --output-root")
        prepare_score_worksets(args.study, jobs, args.output_root)
    elif args.phase == "publish":
        publish(args.study, build)
    else:
        stages = [(j, s, l) for j in jobs for s in j.get("subsets", ("test", "valid")) for l in j["levels"]]
        pending = []
        for job, subset, level in stages:
            if args.phase == "reuse" and job["role"] != "l2plus":
                continue
            path = REPO_ROOT / job["output"] / job["task"] / "scaffold" / subset / level / "VERSION.json"
            if args.ready_only and not path.is_file():
                continue
            manifest = json.loads(path.read_text())
            if args.phase == "reuse" and list((path.parent / ".scores").glob("*.jsonl")):
                continue
            if manifest["status"] == "prepared" or (args.phase == "finalize" and manifest["status"] == "ready_to_finalize"):
                pending.append((job, subset, level))
        if args.phase == "score":
            from queue import Queue, Empty
            queue = Queue()
            for job, subset, level in pending:
                for shard in range(args.num_shards):
                    if args.shard is None or shard in args.shard:
                        queue.put((job, subset, level, shard))
            def worker(device):
                while True:
                    try:
                        job, subset, level, shard = queue.get_nowait()
                    except Empty:
                        return
                    invoke(args.study, job, "score", subset, level, ["--num-shards", str(args.num_shards),
                           "--shard-index", str(shard), "--device", device, "--batch-size", str(args.batch_size),
                           "--gpu-memory-fraction", str(args.gpu_memory_fraction)])
            with ThreadPoolExecutor(max_workers=len(args.gpus.split(","))) as pool:
                list(pool.map(worker, args.gpus.split(",")))
        else:
            action = {"tokenize": "pretokenize", "reuse": "reuse-scores"}.get(args.phase, "finalize")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(lambda item: invoke(args.study, *item[:1], action, *item[1:],
                     ["--num-shards", str(args.num_shards), "--tokenized-root", str(args.study / "tokenized" / item[0]["id"])]
                     if action == "pretokenize" else []), pending))


if __name__ == "__main__":
    main()
