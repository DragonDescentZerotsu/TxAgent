"""Independent validation selection and fixed-split repeated-run reporting."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from scipy.stats import t
import torch

from predict.baselines.minimol import run_multitask as training
from predict.baselines.minimol.condition_features import append_condition_one_hot, condition_vocabulary
from predict.baselines.minimol.head_runtime import TaskHead
from predict.baselines.minimol.run_bioavailability_ma import load_split, choose_threshold, evaluate_metrics
from predict.utils.json import atomic_output_path, sha256_file, write_json_atomic, write_jsonl_atomic

PROTOCOL = dict(schema="independent_seed_evaluation.v1", seeds=[1, 2, 3, 4, 5],
    checkpoint="minimum validation BCE within each seed and lambda",
    auxiliary_lambda="minimum validation BCE within each seed; tie: smaller lambda",
    threshold="maximum validation Macro-F1 within each seed",
    summary="arithmetic mean; sample SD (ddof=1); two-sided Student t 95% CI of mean",
    uncertainty="training randomness conditional on the frozen data split; not test-sampling or split uncertainty",
    test_used_for_selection=False)

FOREST_PROTOCOL = dict(PROTOCOL,
    checkpoint="minimum validation BCE hyperparameter candidate within each seed; tie: candidate path",
    auxiliary_lambda="not applicable; direct/auxiliary retrieval features are separate settings")


def select_auxiliary_policy(source, output):
    """Freeze one MiniMol-derived lambda per pool/task, using validation only."""
    cells = {}
    for folder in sorted((source / "minimol").glob("*/*/indirect")):
        records, sources = [], []
        for seed in PROTOCOL["seeds"]:
            path = folder / f"seed_{seed}" / "metrics.json"
            record = json.loads(path.read_text())
            if record["seed"] != seed or record["protocol"] != PROTOCOL:
                raise ValueError(f"Unexpected MiniMol selection protocol: {path}")
            records.append(record)
            sources.append(dict(path=str(path.resolve()), sha256=sha256_file(path)))
        means = {w: float(np.mean([r["candidate_validation_bce"][w] for r in records]))
                 for w in ("0.1", "0.6", "1.0")}
        if not np.isfinite(list(means.values())).all():
            raise ValueError("Nonfinite MiniMol validation losses")
        cells["/".join(folder.parts[-3:-1])] = dict(
            weight=float(min(means, key=lambda w: (means[w], float(w)))),
            mean_validation_bce=means, sources=sources)
    if len(cells) != 18:
        raise ValueError("Expected 18 MiniMol pool/task cohorts")
    policy = dict(schema="minimol_fixed_auxiliary.v1", seeds=PROTOCOL["seeds"],
                  selection="minimum five-seed mean MiniMol validation BCE per pool/task; tie: smaller lambda",
                  test_used=False, cells=cells)
    if output.exists() and json.loads(output.read_text()) != policy:
        raise ValueError("Fixed auxiliary policy already binds different source results")
    write_json_atomic(output, policy)


def fixed_auxiliary_protocol(policy, cell):
    entry = policy["cells"][cell]
    means = entry["mean_validation_bce"]
    chosen = float(min(means, key=lambda w: (means[w], float(w))))
    if (policy["schema"] != "minimol_fixed_auxiliary.v1" or policy["test_used"]
            or policy["seeds"] != PROTOCOL["seeds"] or entry["weight"] != chosen
            or set(means) != {"0.1", "0.6", "1.0"} or not np.isfinite(list(means.values())).all()):
        raise ValueError("Invalid fixed auxiliary policy")
    return dict(PROTOCOL, auxiliary_lambda=policy["selection"], fixed_auxiliary_weight=chosen)


def prepare_legacy(args):
    table = args.source_root / "outputs/paper/llm_vs_standard_baselines_20260924/ml_appendix.csv"
    methods = {"minimol_head": "frozen", "D1": "frozen", "finetuned": "finetuned", "indirect": "indirect",
        "tdc_only_finetuned": "finetuned", "tdc_only_indirect": "indirect",
        "mixed_finetuned": "finetuned", "mixed_indirect": "indirect"}
    jobs, cells = [], []
    for row in csv.DictReader(table.open()):
        if row["method"] not in methods:
            continue
        setting = methods[row["method"]]
        old = (args.source_root / row["predictions"]).parent
        metrics = json.loads((old / "metrics.json").read_text())
        train_root = old if setting == "frozen" else (args.source_root / metrics["checkpoints"][0]).parents[2]
        prepared = train_root / "prepared"
        if setting != "frozen":
            for weight in ([.1, .6, 1.] if setting == "indirect" else [0.]):
                for seed in PROTOCOL["seeds"]:
                    for file in ("best.pt", "training.json", "valid_scores.npy"):
                        if not (train_root / f"aux_{weight:g}" / f"seed_{seed}" / file).exists():
                            raise FileNotFoundError(f"Missing historical candidate: {train_root}, {weight}, {seed}, {file}")
        elif not (old / "best_ensemble.pt").exists():
            raise FileNotFoundError(old / "best_ensemble.pt")
        cell = dict(model="minimol", training_pool=row["training_pool"], task=row["task"], setting=setting,
                    original_metrics=str(old / "metrics.json"), original_metrics_sha256=sha256_file(old / "metrics.json"),
                    training_root=str(train_root))
        cells.append(cell)
        for seed in PROTOCOL["seeds"]:
            output = args.output / "minimol" / row["training_pool"] / row["task"] / setting / f"seed_{seed}"
            cmd = [str(args.legacy_python), "-m", "predict.baselines.repeated_runs", "legacy-head" if setting == "frozen" else "evaluate",
                   "--source-root", str(args.source_root), "--training-root", str(train_root), "--output", str(output),
                   "--seed", str(seed), "--setting", setting]
            if setting != "frozen":
                cmd += ["--prepared", str(prepared)]
            cmd += ["--device", "cpu"]
            jobs.append(dict(id="legacy_repeated/"+str(output.relative_to(args.output)), dependencies=[],
                marker=str(output / "metrics.json"), command=cmd, cpu=True,
                env=dict(TXAGENT_MINIMOL_RUNTIME_ROOT="/data1/tianang/cache/txagent/minimol",
                         PYTHONPATH="/data1/tianang/cache/txagent/minimol/pydeps:/data1/tianang/cache/txagent/minimol/runtime_41642328")))
    if len(cells) != 54:
        raise ValueError("Expected all 54 historical MiniMol cells")
    manifest = dict(protocol=PROTOCOL, source_table=str(table), source_table_sha256=sha256_file(table),
                    cells=cells, jobs=jobs, output=str(args.output), expected_cells=54)
    target = args.output / "manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("Historical replay identity already exists with different inputs")
    write_json_atomic(target, manifest)


def prepare_forests(args):
    """Locate all retained RASAR members through the frozen paper source table."""
    table = args.source_root / "outputs/paper/llm_vs_standard_baselines_20260924/ml_appendix.csv"
    direct = json.loads((args.source_root / "outputs/paper/retrieval_forest_direct_20260925/manifest.json").read_text())
    prepared_paths = {args.source_root / c["prepared"] for c in direct["cells"]}
    rows = [r for r in csv.DictReader(table.open()) if "rf_" in r["method"]]
    for row in rows:
        original = (args.source_root / row["predictions"]).parent
        prepared_paths.update(p / "prepared" for p in (original, original.parent, original.parent.parent)
                              if (p / "prepared/manifest.json").exists())
    prepared_by_hash = {sha256_file(p / "manifest.json"): p for p in sorted(prepared_paths)}
    cells, jobs = [], []
    for row in rows:
        original = (args.source_root / row["predictions"]).parent
        metrics_path = original / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        candidate = json.loads(next((original / "candidates").glob("*/training.json")).read_text())
        prepared = prepared_by_hash[metrics["prepared_sha256"]]
        feature_set = candidate.get("feature_set", "direct_indirect")
        setting = ("indirect" if feature_set == "direct_indirect" else "direct") + (
            "_condition" if row["method"].endswith("_condition") else "_all")
        cells.append(dict(model="rasar", training_pool=row["training_pool"], task=row["task"], setting=setting,
                          original_metrics=str(metrics_path), original_metrics_sha256=sha256_file(metrics_path)))
        for seed in PROTOCOL["seeds"]:
            output = args.output / "rasar" / row["training_pool"] / row["task"] / setting / f"seed_{seed}"
            jobs.append(dict(id=str(output.relative_to(args.output)), marker=str(output / "metrics.json"),
                command=[str(args.legacy_python), "-m", "predict.baselines.repeated_runs", "legacy-forest",
                    "--training-root", str(original), "--prepared", str(prepared),
                    "--data-dir", str(args.source_root / json.loads((prepared / "manifest.json").read_text())["data_dir"]),
                    "--policy", str(candidate["policy"]), "--feature-set", feature_set,
                    "--seed", str(seed), "--output", str(output)]))
    if len(cells) != 72:
        raise ValueError("Expected 72 historical RASAR cells")
    manifest = dict(protocol=FOREST_PROTOCOL, source_table=str(table), source_table_sha256=sha256_file(table),
                    cells=cells, jobs=jobs, output=str(args.output), expected_cells=72)
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Historical forest replay already binds different inputs")
    write_json_atomic(path, manifest)


def legacy_forest(args):
    from predict.baselines.retrieval_forest import validated_candidates, load_features
    import joblib
    if (args.output / "metrics.json").exists():
        raise ValueError("Completed forest replay exists")
    manifest, features, source_rows = load_features(args.prepared, args.policy, args.feature_set, data_dir=args.data_dir)
    candidates = validated_candidates(SimpleNamespace(**{**vars(args), "output": args.training_root,
                                                         "expected_candidates": 6}), manifest, source_rows)
    labels = np.asarray([r["Y"] for r in source_rows["valid"]])
    members = {}
    for _, path, receipt, scores in candidates:
        seeds = [m["seed"] for m in receipt["members"]]
        if sorted(seeds) != PROTOCOL["seeds"]:
            raise ValueError("Expected five distinct saved forest seeds")
        i = seeds.index(args.seed)
        members[path] = dict(member=receipt["members"][i], scores=scores[i],
                             bce=training.ensemble_bce(labels, scores[i]))
    selected = min(members, key=lambda p: (members[p]["bce"], p))
    winner = members[selected]
    model = joblib.load(winner["member"]["path"])
    model.n_jobs = 1
    valid = model.predict_proba(features["valid"])[:, 1]
    if not np.allclose(valid, winner["scores"], rtol=1e-7, atol=1e-9):
        raise ValueError("Replayed forest validation scores differ from saved candidate")
    threshold, _ = choose_threshold(labels.tolist(), valid, "valid_macro_f1")
    scores = dict(valid=valid, test=model.predict_proba(features["test"])[:, 1])
    rows = {s: [dict(index=i, drug=r["drug"], label=r["Y"],
                    condition_group=r.get("condition_group", r.get("model_condition")))
                for i, r in enumerate(source_rows[s])] for s in ("valid", "test")}
    save_evaluation(args.output, rows, scores, seed=args.seed, weight=0, threshold=threshold,
        candidates={0: winner["bce"]}, protocol=FOREST_PROTOCOL,
        provenance=dict(encoder="rasar", precision="sklearn", selected_candidate=selected,
            candidate_validation_bce={p: m["bce"] for p, m in members.items()},
            checkpoint=winner["member"], prepared=str(args.prepared),
            prepared_sha256=sha256_file(args.prepared / "manifest.json"),
            policy=args.policy, feature_set=args.feature_set))


def run_legacy(args):
    manifest = json.loads((args.output / "manifest.json").read_text())
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    pending = [j for j in manifest["jobs"] if not Path(j["marker"]).exists()]
    def replay(job):
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env.update(job.get("env", {}))
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        with (logs / (job["id"].replace("/", "__") + ".log")).open("a") as log:
            code = subprocess.run(job["command"], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        return job["id"] if code or not Path(job["marker"]).exists() else None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        failed = [job for job in pool.map(replay, pending) if job is not None]
    if failed:
        raise RuntimeError(f"Replay failed; inspect logs and resume: {failed}")
    summarize(args.output, manifest["expected_cells"])


def statistics(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Repeated-run statistics require at least two finite scores")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    radius = float(t.ppf(.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return dict(n=len(values), mean=mean, sd=sd, ci95_low=mean-radius, ci95_high=mean+radius)


def select_member(members, labels):
    losses = {weight: training.ensemble_bce(labels, member["scores"]) for weight, member in members.items()}
    weight = min(losses, key=lambda w: (losses[w], w))
    threshold, _ = choose_threshold(labels.tolist(), members[weight]["scores"], "valid_macro_f1")
    return weight, threshold, losses


def save_evaluation(output, rows, scores, *, seed, weight, threshold, candidates, provenance, protocol=None):
    output.mkdir(parents=True, exist_ok=True)
    metrics = dict(protocol=protocol or PROTOCOL, seed=seed, selected_aux_weight=weight, threshold=threshold,
                   candidate_validation_bce=candidates, provenance=provenance, prediction_sha256={})
    for split in ("valid", "test"):
        labels = [r["label"] for r in rows[split]]
        if len(scores[split]) != len(labels) or not np.isfinite(scores[split]).all():
            raise ValueError("Incomplete or nonfinite predictions")
        predictions = [dict(row, probability=float(p), prediction=int(p >= threshold))
                       for row, p in zip(rows[split], scores[split])]
        file = output / f"{split}_predictions.jsonl"
        write_jsonl_atomic(file, predictions)
        metrics["prediction_sha256"][split] = sha256_file(file)
        metrics[split] = evaluate_metrics(labels, scores[split], threshold)
    write_json_atomic(output / "metrics.json", metrics)


def evaluate(args):
    """Evaluate one seed; all candidate selection is completed before test inference."""
    if (args.output / "metrics.json").exists():
        raise ValueError("Completed repeated-run evaluation exists")
    torch.set_num_threads(2)
    data, graphs = training.load_prepared(args.prepared, with_graphs=False, encoder=args.encoder)
    labels = data["splits"]["valid"]["labels"].numpy()
    weights = [.1, .6, 1.] if args.setting == "indirect" else [0.]
    protocol = PROTOCOL
    if getattr(args, "auxiliary_policy", None):
        if args.setting != "indirect" or not args.auxiliary_cell:
            raise ValueError("Fixed auxiliary policy requires an indirect pool/task evaluation")
        protocol = fixed_auxiliary_protocol(json.loads(args.auxiliary_policy.read_text()), args.auxiliary_cell)
        weights = [protocol["fixed_auxiliary_weight"]]
    binding = (dict(features_sha256=sha256_file(args.feature_cache / "receipt.json"),
                    encoder=args.encoder, frozen=args.setting == "frozen") if args.encoder != "minimol" else {})
    members = {w: training.validated_member(args.training_root / f"aux_{w:g}" / f"seed_{args.seed}",
                prepared_sha256=sha256_file(args.prepared / "manifest.json"), seed=args.seed,
                weight=w, labels=labels, allow_bce_replay=len(weights) == 1, **binding) for w in weights}
    signatures = {tuple(m["config"].get(k) for k in
                  ("hidden_dim", "depth", "lr", "encoder_lr", "epochs", "batch_size", "precision"))
                  for m in members.values()}
    if len(signatures) != 1:
        raise ValueError("Candidate configurations differ beyond auxiliary lambda")
    weight, threshold, losses = select_member(members, labels)
    member = members[weight]
    checkpoint = torch.load(member["path"] / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    encoder = None
    if args.encoder != "minimol":
        from predict.baselines.foundation_models import FoundationEncoder
        encoder = FoundationEncoder(args.encoder, args.runtime_root, args.feature_cache, data["smiles"],
                                    frozen=args.setting == "frozen")
    model = training.MultiTaskModel(len(data["conditions"]), len(data["buckets"]) if weight else 0,
                                   config["hidden_dim"], config["depth"], encoder=encoder)
    model.load_state_dict(checkpoint["state_dict"])
    model.precision = config.get("precision", "fp32")
    model.to(args.device)
    if graphs is None:
        graphs = torch.load(args.prepared / "graphs.pt", weights_only=False)
    batch_size = 32 if args.encoder == "unimol" else 128
    replay_receipt = None
    if member["validation_replay_required"]:
        # Reproduce the training validator's CUDA math, then restore the normal
        # evaluation settings before any test inference.
        matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            replay_scores, replay_bce = training.predict(model, training.loader(graphs,
                data["splits"]["valid"], config.get("eval_batch_size", batch_size)), args.device)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul_tf32, cudnn_tf32
        if (not np.allclose(replay_scores, member["scores"], rtol=1e-4, atol=1e-5)
                or not np.isclose(replay_bce, member["checkpoint_validation_bce"], rtol=1e-4, atol=1e-5)):
            raise ValueError(f"Checkpoint validation replay mismatch: logit BCE {replay_bce} vs "
                f"{member['checkpoint_validation_bce']}; max probability difference "
                f"{np.max(np.abs(replay_scores-member['scores']))}")
        replay_receipt = dict(logit_bce=replay_bce, checkpoint_logit_bce=member["checkpoint_validation_bce"],
            probability_bce=losses[weight], max_probability_difference=float(np.max(np.abs(replay_scores-member["scores"]))),
            replay_rtol=1e-4, replay_atol=1e-5,
            reason="Float32 probability rounding; hash-bound scores and checkpoint logit BCE independently replayed")
    scores = dict(valid=member["scores"], test=training.predict(model,
        training.loader(graphs, data["splits"]["test"], batch_size), args.device)[0])
    rows = {split: [dict(index=i, drug=data["smiles"][int(index)], label=int(target),
                        condition_group=data["splits"][split].get("condition_groups", [None]*len(values["indices"]))[i])
                    for i, (index, target) in enumerate(zip(values["indices"], values["labels"]))]
            for split, values in data["splits"].items() if split in ("valid", "test")}
    provenance = dict(encoder=args.encoder, precision=model.precision, inference_device=args.device,
        prepared=str(args.prepared), prepared_sha256=sha256_file(args.prepared / "manifest.json"),
        selected_checkpoint=str(member["path"] / "best.pt"),
        candidates={w: dict(checkpoint=str(m["path"] / "best.pt"), checkpoint_sha256=m["checkpoint_sha256"],
                            valid_scores_sha256=m["valid_scores_sha256"]) for w, m in members.items()})
    if getattr(args, "auxiliary_policy", None):
        provenance.update(auxiliary_policy=str(args.auxiliary_policy.resolve()),
                          auxiliary_policy_sha256=sha256_file(args.auxiliary_policy), auxiliary_cell=args.auxiliary_cell)
    if replay_receipt is not None:
        provenance["validation_replay"] = replay_receipt
    save_evaluation(args.output, rows, scores, seed=args.seed, weight=weight, threshold=threshold,
                    candidates=losses, provenance=provenance, protocol=protocol)


def legacy_head(args):
    """Unpack a historical five-head checkpoint without refitting its models."""
    if (args.output / "metrics.json").exists():
        raise ValueError("Completed repeated-run evaluation exists")
    source = args.training_root
    checkpoint_path = source / "best_ensemble.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["hparams"]
    data_dir = args.source_root / config["data_dir"]
    splits = {s: load_split(data_dir / f"{s}.jsonl", condition_field=config.get("condition_field"))
              for s in ("train", "valid", "test")}
    vocabulary = condition_vocabulary(splits["train"].conditions) if splits["train"].conditions is not None else None
    if (vocabulary or []) != checkpoint["condition_features"]["vocabulary"]:
        raise ValueError("Historical condition vocabulary changed")
    results = checkpoint["run_results"]
    index = [r["seed"] for r in results].index(args.seed)
    state = checkpoint["model_state_dicts"][index]
    model = TaskHead(input_dim=checkpoint["condition_features"]["model_input_dim"],
                     hidden_dim=config["hidden_dim"], depth=config["depth"], dropout=config["dropout"])
    model.load_state_dict(state)
    model.to(args.device).eval()
    scores, rows, files = {}, {}, {}
    for split in ("valid", "test"):
        cache = source / "embeddings" / f"{split}.pt"
        if not cache.exists() and config.get("embedding_cache_dir"):
            cache = args.source_root / config["embedding_cache_dir"] / f"{split}.pt"
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        expected = splits[split]
        if payload["smiles"] != expected.smiles or list(payload["labels"]) != expected.labels:
            raise ValueError(f"Historical embedding/split identity changed: {cache}")
        features = append_condition_one_hot(payload["embeddings"].float(), expected.conditions, vocabulary,
                                           unseen_policy=config.get("unseen_condition_policy", "error"))
        with torch.no_grad():
            scores[split] = torch.cat([model(x.to(args.device)).flatten().sigmoid().cpu()
                                      for x in features.split(128)]).numpy()
        rows[split] = [dict(index=i, drug=s, label=y, condition_group=(expected.conditions or [None]*len(expected.labels))[i])
                       for i, (s, y) in enumerate(zip(expected.smiles, expected.labels))]
        files[split] = dict(split_sha256=sha256_file(data_dir / f"{split}.jsonl"), embedding_sha256=sha256_file(cache))
    bce = training.ensemble_bce(np.asarray(splits["valid"].labels), scores["valid"])
    if not np.isclose(bce, results[index]["best_valid_loss"], rtol=1e-4, atol=1e-5):
        raise ValueError("Replayed frozen head disagrees with historical validation BCE")
    threshold, _ = choose_threshold(splits["valid"].labels, scores["valid"], "valid_macro_f1")
    save_evaluation(args.output, rows, scores, seed=args.seed, weight=0, threshold=threshold,
        candidates={0: bce}, provenance=dict(encoder="minimol", precision="fp32", inference_device=args.device, checkpoint=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path), member_index=index, inputs=files,
        original_metrics_sha256=sha256_file(source / "metrics.json")))


def summarize(root, expected_cells, *, sources=None):
    """Recompute all reported metrics and selection thresholds before aggregation."""
    cells, per_seed, seen = [], [], set()
    for source in sources or [root]:
        if (source / "manifest.json").exists():
            manifest = json.loads((source / "manifest.json").read_text())
            if "source_table_sha256" in manifest:
                if sha256_file(Path(manifest["source_table"])) != manifest["source_table_sha256"]:
                    raise ValueError("Historical ensemble table changed")
                for cell in manifest["cells"]:
                    if sha256_file(Path(cell["original_metrics"])) != cell["original_metrics_sha256"]:
                        raise ValueError("Historical ensemble metrics changed")
    folders = [(source, folder) for source in (sources or [root]) for folder in sorted(source.glob("*/*/*/*"))]
    for source, folder in folders:
        if not folder.is_dir():
            continue
        model, pool, task, setting = folder.relative_to(source).parts
        key = (model, pool, task, setting)
        if key in seen:
            raise ValueError(f"Duplicate result cell across sources: {key}")
        seen.add(key)
        members = []
        identities = {}
        for seed in PROTOCOL["seeds"]:
            path = folder / f"seed_{seed}"
            if not (path / "metrics.json").exists():
                continue
            record = json.loads((path / "metrics.json").read_text())
            protocol = FOREST_PROTOCOL if model == "rasar" else PROTOCOL
            if model == "rasar":
                selection = record["provenance"]
                candidates = selection["candidate_validation_bce"]
                chosen = min(candidates, key=lambda p: (candidates[p], p))
                if (selection["selected_candidate"] != chosen or not np.isclose(
                        candidates[chosen], record["candidate_validation_bce"]["0"])):
                    raise ValueError(f"Forest candidate selection mismatch: {path}")
            policy_path = source / "auxiliary_policy.json"
            if policy_path.exists() and setting == "indirect":
                policy = json.loads(policy_path.read_text())
                protocol = fixed_auxiliary_protocol(policy, f"{pool}/{task}")
                if (record["provenance"].get("auxiliary_policy_sha256") != sha256_file(policy_path)
                        or record["provenance"].get("auxiliary_cell") != f"{pool}/{task}"
                        or set(record["candidate_validation_bce"]) != {str(protocol["fixed_auxiliary_weight"])}):
                    raise ValueError(f"Fixed auxiliary provenance mismatch: {path}")
            if record["protocol"] != protocol or record["seed"] != seed:
                raise ValueError(f"Evaluation protocol mismatch: {path}")
            losses = {float(k): v for k, v in record["candidate_validation_bce"].items()}
            chosen = min(losses, key=lambda w: (losses[w], w))
            if chosen != record["selected_aux_weight"]:
                raise ValueError(f"Lambda selection mismatch: {path}")
            for split in ("valid", "test"):
                file = path / f"{split}_predictions.jsonl"
                if sha256_file(file) != record["prediction_sha256"][split]:
                    raise ValueError(f"Changed predictions: {file}")
                rows = [json.loads(line) for line in file.read_text().splitlines()]
                identity = [(r["index"], r["drug"], r["label"], r.get("condition_group")) for r in rows]
                if identity != identities.setdefault(split, identity):
                    raise ValueError(f"Different evaluation cohort across seeds: {file}")
                y, p = [r["label"] for r in rows], np.array([r["probability"] for r in rows])
                if [r["index"] for r in rows] != list(range(len(rows))) or any(
                        r["prediction"] != int(r["probability"] >= record["threshold"]) for r in rows):
                    raise ValueError(f"Prediction ordering/threshold mismatch: {file}")
                if split == "valid":
                    threshold, _ = choose_threshold(y, p, "valid_macro_f1")
                    if threshold != record["threshold"] or not np.isclose(training.ensemble_bce(np.array(y), p), losses[chosen]):
                        raise ValueError(f"Validation selection mismatch: {path}")
                actual = evaluate_metrics(y, p, record["threshold"])
                if any(not np.isclose(v, record[split][k], equal_nan=True) for k, v in actual.items()):
                    raise ValueError(f"Metric mismatch: {file}")
            member = dict(model=model, training_pool=pool, task=task, setting=setting, seed=seed,
                          selected_aux_weight=chosen, **record["test"],
                          precision=record["provenance"]["precision"])
            per_seed.append(member)
            members.append(member)
        if len(members) == 5:
            if len({m["precision"] for m in members}) != 1:
                raise ValueError(f"Mixed precision within a repeated-run cell: {folder}")
            row = dict(model=model, training_pool=pool, task=task, setting=setting,
                       n_seeds=5, precision=members[0]["precision"])
            for metric in ("macro_f1", "auroc", "accuracy"):
                row.update({f"{metric}_{k}": v for k, v in statistics([m[metric] for m in members]).items() if k != "n"})
            cells.append(row)
    for name, rows in (("scores.csv", cells), ("per_seed.csv", per_seed)):
        if rows:
            with atomic_output_path(root / name) as temporary:
                with temporary.open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json_atomic(root / "results.json", cells)
    policies, protocols = {}, {}
    for source in sources or [root]:
        policy, manifest = source / "auxiliary_policy.json", source / "manifest.json"
        protocol = json.loads(manifest.read_text()).get("protocol", PROTOCOL) if manifest.exists() else PROTOCOL
        policies[str(source)] = json.loads(policy.read_text())["selection"] if policy.exists() else protocol["auxiliary_lambda"]
        protocols[str(source)] = dict(protocol, auxiliary_lambda=policies[str(source)])
    write_json_atomic(root / "verification.json", dict(protocol=dict(PROTOCOL,
        checkpoint="source-specific; see selection_protocol_by_source",
        auxiliary_lambda="source-specific; see auxiliary_selection_by_source"), auxiliary_selection_by_source=policies,
        selection_protocol_by_source=protocols,
        verified_cells=len(cells),
        evaluated_seeds=len(per_seed), expected_cells=expected_cells, complete=len(cells) == expected_cells,
        sources=[str(p) for p in (sources or [root])]))
    lines = ["# Independent-seed baseline results", "", "Mean ± sample SD [95% t CI of the mean]; five seeds on the fixed split.",
             "These intervals quantify training randomness, not test-sampling or split uncertainty. Historical ensembles are retained separately.",
             "Auxiliary selection follows each source's protocol and auxiliary_policy.json when present; MiniMol legacy results retain per-seed selection.",
             "", "| Model | Pool | Task | Setting | Precision | Macro-F1 | AUROC |", "|---|---|---|---|---|---|---|"]
    for row in cells:
        values = [f"{row[m+'_mean']:.4f} ± {row[m+'_sd']:.4f} [{row[m+'_ci95_low']:.4f}, {row[m+'_ci95_high']:.4f}]"
                  for m in ("macro_f1", "auroc")]
        lines.append("| " + " | ".join([row[k] for k in ("model", "training_pool", "task", "setting", "precision")] + values) + " |")
    with atomic_output_path(root / "MAIN_TABLE.md") as temporary:
        temporary.write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(verified_cells=len(cells), evaluated_seeds=len(per_seed))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("evaluate", "legacy-head", "legacy-forest", "summarize", "prepare-legacy", "prepare-forests", "run-legacy", "select-auxiliary"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training-root", type=Path)
    p.add_argument("--prepared", type=Path)
    p.add_argument("--encoder", default="minimol")
    p.add_argument("--runtime-root", type=Path)
    p.add_argument("--feature-cache", type=Path)
    p.add_argument("--source-root", type=Path, default=Path('/data1/tianang/Projects/TxAgent'))
    p.add_argument("--legacy-python", type=Path, default=Path('/data1/tianang/anaconda3/envs/vllm/bin/python'))
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--policy", type=int, choices=(0, 1))
    p.add_argument("--feature-set", choices=("direct", "direct_indirect"))
    p.add_argument("--sources", type=Path, nargs="+", help="Distinct repeated-run report roots to combine")
    p.add_argument("--workers", type=int, default=16, help="Concurrent CPU checkpoint replay processes")
    p.add_argument("--setting", choices=("frozen", "finetuned", "indirect"))
    p.add_argument("--seed", type=int, choices=range(1, 6))
    p.add_argument("--device", default="cuda")
    p.add_argument("--expected-cells", type=int, default=378)
    p.add_argument("--auxiliary-policy", type=Path)
    p.add_argument("--auxiliary-cell", help="Pool/task key in the fixed auxiliary policy")
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.phase == "select-auxiliary":
        select_auxiliary_policy(args.source_root, args.output)
    elif args.phase == "summarize":
        summarize(args.output, args.expected_cells, sources=args.sources)
    elif args.phase == "prepare-legacy":
        prepare_legacy(args)
    elif args.phase == "prepare-forests":
        prepare_forests(args)
    elif args.phase == "run-legacy":
        run_legacy(args)
    else:
        torch.set_num_threads(2)
        {"evaluate": evaluate, "legacy-head": legacy_head, "legacy-forest": legacy_forest}[args.phase](args)


if __name__ == "__main__":
    main()
