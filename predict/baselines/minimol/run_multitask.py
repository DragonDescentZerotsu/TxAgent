"""Prepare, embed, train and evaluate parent-disjoint MiniMol auxiliary tasks."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from predict.baselines.minimol.finetune_runtime import (
    GraphRows, MiniMolEncoder, auxiliary_loss, collate_graphs,
)
from predict.baselines.minimol.head_runtime import TaskHead
from predict.baselines.minimol.run_bioavailability_ma import choose_threshold, evaluate_metrics, set_seed
from predict.utils.json import sha256_file, write_json_atomic


class MultiTaskModel(nn.Module):
    def __init__(self, conditions: int, buckets: int, hidden_dim: int, depth: int, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else MiniMolEncoder()
        width = getattr(self.encoder, "output_dim", 512)
        self.head = TaskHead(input_dim=width + conditions, hidden_dim=hidden_dim, depth=depth)
        self.auxiliary = nn.Linear(width, buckets) if buckets else None

    def main_logits(self, embeddings, conditions):
        return self.head(torch.cat((embeddings, conditions), dim=-1)).squeeze(-1)


def loader(graphs, rows, batch_size, *, workers=0, shuffle=False, sampler=None, auxiliary=False):
    return DataLoader(
        GraphRows(graphs, rows, auxiliary), batch_size=batch_size,
        shuffle=shuffle, sampler=sampler, collate_fn=collate_graphs,
        num_workers=workers, persistent_workers=workers > 0,
        pin_memory=True,
    )


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    logits, labels = [], []
    for graph, conditions, target in batches:
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                            enabled=getattr(model, "precision", "fp32") == "bf16"):
            embeddings = model.encoder(graph.to(getattr(model.encoder, "input_device", device), non_blocking=True))
            output = model.main_logits(embeddings, conditions.to(device, non_blocking=True))
        logits.append(output.float().cpu())
        labels.append(target)
    logits, labels = torch.cat(logits), torch.cat(labels)
    return logits.sigmoid().numpy(), float(nn.functional.binary_cross_entropy_with_logits(logits, labels))


def load_prepared(root, *, with_graphs=True, encoder="minimol"):
    manifest = json.loads((root / "manifest.json").read_text())
    data_path = root / "data.pt"
    if encoder != "minimol":
        from predict.baselines.foundation_cache import cached_input
        data_path = cached_input(data_path, manifest["data_sha256"])
    if sha256_file(data_path) != manifest["data_sha256"]:
        raise ValueError("Prepared target cache hash changed")
    if encoder != "minimol":
        data = torch.load(data_path, weights_only=False)
        return data, torch.arange(len(data["smiles"]))
    graph_receipt = json.loads((root / "graphs.json").read_text())
    if graph_receipt["data_sha256"] != manifest["data_sha256"]:
        raise ValueError("Graph and target caches are from different inputs")
    if sha256_file(root / "graphs.pt") != graph_receipt["graph_sha256"]:
        raise ValueError("Prepared graph cache hash changed")
    data = torch.load(root / "data.pt", weights_only=False)
    graphs = torch.load(root / "graphs.pt", weights_only=False) if with_graphs else None
    if graph_receipt["n_graphs"] != len(data["smiles"]) or (graphs is not None and len(graphs) != len(data["smiles"])):
        raise ValueError("Graph/SMILES alignment mismatch")
    return data, graphs


def prepare(args):
    from predict.baselines.minimol.indirect_data import prepare as prepare_targets
    from predict.baselines.minimol.finetune_runtime import prepare_graphs

    if args.prepared.exists() and any(args.prepared.iterdir()):
        raise ValueError("Preparation requires an empty directory; preserve existing run inputs")
    prepare_targets(args.data_dir, args.auxiliary, args.prepared, args.prepare_workers,
                    condition_field=args.condition_field, preserve_main_split=args.preserve_main_split)
    prepare_graphs(args.prepared / "data.pt", args.prepared / "graphs.pt", args.prepare_workers)


@torch.no_grad()
def embed(args):
    """Export exactly the same graph features for the existing frozen-head runner."""
    torch.set_num_threads(args.threads)
    data, graphs = load_prepared(args.prepared)
    model = MiniMolEncoder().to(args.device).eval()
    output = args.output / "embeddings"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.pt")):
        raise ValueError("Embedding output already exists; use the existing cache or a new output")
    for split, rows in data["splits"].items():
        batches = loader(graphs, rows, args.eval_batch_size, workers=args.workers)
        features = torch.cat([model(g.to(args.device)).cpu() for g, *_ in batches])
        if not torch.isfinite(features).all():
            raise FloatingPointError("Nonfinite frozen embeddings")
        torch.save(dict(smiles=[data["smiles"][int(i)] for i in rows["indices"]],
                        labels=rows["labels"].tolist(), embeddings=features), output / f"{split}.pt")
    write_json_atomic(output / "manifest.json", dict(
        prepared_sha256=sha256_file(args.prepared / "manifest.json"),
        graph_sha256=sha256_file(args.prepared / "graphs.pt"),
        files={p.name: sha256_file(p) for p in sorted(output.glob("*.pt"))},
    ))


def train(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "training.json").exists():
        raise ValueError("Completed training exists; use a new output directory")
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    data, graphs = load_prepared(args.prepared, encoder=getattr(args, "encoder", "minimol"))
    device = torch.device(args.device)
    encoder = None
    frozen = getattr(args, "frozen", False)
    if getattr(args, "encoder", "minimol") != "minimol":
        from predict.baselines.foundation_models import FoundationEncoder
        encoder = FoundationEncoder(args.encoder, args.runtime_root, args.feature_cache, data["smiles"], frozen=frozen)
        # Some official checkpoint loaders restore their pretraining RNG state.
        set_seed(args.seed)
    model = MultiTaskModel(len(data["conditions"]), len(data["buckets"]) if args.aux_weight else 0,
                           args.hidden_dim, args.depth, encoder=encoder).to(device)
    model.precision = getattr(args, "precision", "fp32")
    # Initialization is identical across aux weights for a given seed.
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = loader(graphs, data["splits"]["train"], args.batch_size, workers=args.workers, shuffle=True)
    train_loader.generator = generator
    train_loader.sampler.generator = generator
    valid_loader = loader(graphs, data["splits"]["valid"], args.eval_batch_size, workers=args.workers)
    auxiliary_loader = None
    if args.aux_weight:
        aux = data["auxiliary"]
        support = torch.bincount(aux["buckets"], minlength=len(data["buckets"]))
        weights = 1.0 / support[aux["buckets"]].double()
        sampler = WeightedRandomSampler(weights, len(train_loader) * args.batch_size,
                                        generator=torch.Generator().manual_seed(args.seed + 10000))
        auxiliary_loader = loader(graphs, aux, args.batch_size, workers=args.workers,
                                  sampler=sampler, auxiliary=True)
    groups = [{"params": model.encoder.parameters(), "lr": args.encoder_lr},
              {"params": model.head.parameters(), "lr": args.lr}]
    if model.auxiliary is not None:
        groups.append({"params": model.auxiliary.parameters(), "lr": args.lr})
    optimizer = torch.optim.Adam(groups, weight_decay=1e-4)

    def schedule(epoch):
        if epoch + 1 <= 5:
            return (epoch + 1) / 5
        return (1 + math.cos(math.pi * min(args.epochs - 5, epoch + 1 - 5) / max(1, args.epochs - 5))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    history, best_loss, best_epoch = [], float("inf"), 0
    gradient_receipt = {}
    started = time.monotonic()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    binding = dict(args=config, prepared_sha256=sha256_file(args.prepared / "manifest.json"))
    if encoder is not None:
        binding["features_sha256"] = sha256_file(args.feature_cache / "receipt.json")
    resume_path = args.output / "resume.pt"
    first_epoch = 1
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        previous = dict(saved["binding"], args={k: v for k, v in saved["binding"]["args"].items() if k != "output"})
        requested = dict(binding, args={k: v for k, v in binding["args"].items() if k != "output"})
        if previous != requested:
            raise ValueError("Resume configuration or frozen inputs changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, best_loss, best_epoch = saved["history"], saved["best_loss"], saved["best_epoch"]
        gradient_receipt = saved["gradient_receipt"]
        first_epoch = saved["epoch"] + 1
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        generator.set_state(saved["loader_rng"])
        if auxiliary_loader is not None:
            auxiliary_loader.sampler.generator.set_state(saved["auxiliary_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        del saved
    for epoch in range(first_epoch, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        totals = np.zeros(2)
        aux_iterator = iter(auxiliary_loader) if auxiliary_loader is not None else None
        for graph, conditions, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            graph = graph.to(getattr(model.encoder, "input_device", device), non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=model.precision == "bf16"):
                main_embeddings = model.encoder(graph)
                logits = model.main_logits(main_embeddings, conditions.to(device, non_blocking=True))
                main_loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.to(device, non_blocking=True))
            # Release main activations before auxiliary forward, keeping gradients accumulated.
            main_loss.backward()
            aux_loss = torch.zeros((), device=device)
            if aux_iterator is not None:
                aux_graph, buckets, targets, binary = next(aux_iterator)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=model.precision == "bf16"):
                    embeddings = model.encoder(aux_graph.to(getattr(model.encoder, "input_device", device), non_blocking=True))
                    aux_logits = model.auxiliary(embeddings).gather(1, buckets.to(device)[:, None]).squeeze(1)
                    aux_loss = auxiliary_loss(aux_logits.float(), targets.to(device), binary.to(device))
                (args.aux_weight * aux_loss).backward()
            if not torch.isfinite(main_loss + aux_loss):
                raise FloatingPointError("Nonfinite loss")
            if not gradient_receipt and not frozen:
                nonzero = sum(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.encoder.parameters())
                if not nonzero:
                    raise RuntimeError("Encoder gradients are disconnected")
                gradient_receipt = dict(encoder_parameters_with_nonzero_gradient=nonzero)
            if not frozen:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            totals += [main_loss.detach().item(), aux_loss.detach().item()]
        scheduler.step()
        scores, valid_bce = predict(model, valid_loader, device)
        if valid_bce < best_loss:
            best_loss, best_epoch = valid_bce, epoch
            checkpoint = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                              args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                              epoch=epoch, valid_bce=valid_bce,
                              prepared_sha256=binding["prepared_sha256"], binding=binding)
            np.save(args.output / "valid_scores.npy", scores)
            checkpoint["valid_scores_sha256"] = sha256_file(args.output / "valid_scores.npy")
            torch.save(checkpoint, args.output / "best.tmp")
            (args.output / "best.tmp").replace(args.output / "best.pt")
        row = dict(epoch=epoch, train_bce=float(totals[0] / len(train_loader)),
                   auxiliary_loss=float(totals[1] / len(train_loader)), valid_bce=valid_bce,
                   seconds=time.monotonic() - epoch_start, best_epoch=best_epoch)
        history.append(row)
        # Epoch-boundary resume preserves sampler and dropout randomness.
        saved = dict(binding=binding, epoch=epoch, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     history=history, best_loss=best_loss, best_epoch=best_epoch,
                     gradient_receipt=gradient_receipt, python_rng=random.getstate(),
                     numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                     loader_rng=generator.get_state(),
                     auxiliary_rng=auxiliary_loader.sampler.generator.get_state() if auxiliary_loader else None,
                     cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
        torch.save(saved, args.output / "resume.tmp")
        (args.output / "resume.tmp").replace(resume_path)
        write_json_atomic(args.output / "progress.json", dict(history=history, best_epoch=best_epoch))
        print(json.dumps(row), flush=True)
    write_json_atomic(args.output / "training.json", dict(
        seed=args.seed, aux_weight=args.aux_weight, best_epoch=best_epoch, best_valid_bce=best_loss,
        history=history, gradient_check=gradient_receipt, elapsed_seconds=time.monotonic() - started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        checkpoint_sha256=sha256_file(args.output / "best.pt"),
        test_used=False,
    ))
    resume_path.unlink(missing_ok=True)


def ensemble_bce(labels, scores):
    p = np.clip(scores.astype(np.float64), 1e-12, 1 - 1e-12)
    return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())


def validated_member(path, *, prepared_sha256, seed, weight, labels, features_sha256=None, encoder="minimol", frozen=False,
                     allow_bce_replay=False):
    """Check every candidate before validation selection, including non-winners."""
    receipt = json.loads((path / "training.json").read_text())
    digest = sha256_file(path / "best.pt")
    if digest != receipt["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint changed: {path}")
    checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    if features_sha256 is not None:
        if (checkpoint.get("binding", {}).get("features_sha256") != features_sha256
                or config.get("encoder") != encoder or config.get("frozen") != frozen):
            raise ValueError(f"Checkpoint encoder/feature/setting mismatch: {path}")
    if checkpoint["prepared_sha256"] != prepared_sha256:
        raise ValueError(f"Checkpoint input manifest mismatch: {path}")
    if config["seed"] != seed or config["aux_weight"] != weight:
        raise ValueError(f"Seed or auxiliary weight mismatch: {path}")
    if receipt["test_used"] or len(receipt["history"]) != config["epochs"]:
        raise ValueError(f"Incomplete or invalid training receipt: {path}")
    best = min(receipt["history"], key=lambda row: row["valid_bce"])
    if best["epoch"] != checkpoint["epoch"] or best["epoch"] != receipt["best_epoch"]:
        raise ValueError(f"Checkpoint was not selected by validation BCE: {path}")
    scores = np.load(path / "valid_scores.npy")
    score_hash = sha256_file(path / "valid_scores.npy")
    if "valid_scores_sha256" in checkpoint and score_hash != checkpoint["valid_scores_sha256"]:
        raise ValueError(f"Validation scores changed: {path}")
    if scores.shape != labels.shape or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"Invalid validation probabilities: {path}")
    if not np.isclose(best["valid_bce"], checkpoint["valid_bce"], rtol=1e-6, atol=1e-7):
        raise ValueError(f"Checkpoint BCE disagrees with training history: {path}")
    # Float32 sigmoid can round to 1, losing the information needed to recover
    # logit BCE. Only hash-bound scores may defer this check to model replay.
    replay = not np.isclose(ensemble_bce(labels, scores), checkpoint["valid_bce"], rtol=1e-4, atol=1e-5)
    if replay and not (allow_bce_replay and "valid_scores_sha256" in checkpoint):
        raise ValueError(f"Validation scores do not match checkpoint BCE: {path}")
    return dict(path=path, config=config, scores=scores, checkpoint_sha256=digest,
                valid_scores_sha256=score_hash, validation_replay_required=bool(replay),
                checkpoint_validation_bce=checkpoint["valid_bce"])


def evaluation_inputs(prepared, members, threshold, *, feature_cache=None):
    return dict(prepared_sha256=sha256_file(prepared / "manifest.json"),
                graph_sha256=(sha256_file(feature_cache / "receipt.json") if feature_cache else
                              json.loads((prepared / "graphs.json").read_text())["graph_sha256"]),
                checkpoint_sha256=[m["checkpoint_sha256"] for m in members],
                valid_scores_sha256=[m["valid_scores_sha256"] for m in members], threshold=threshold)


def reusable_evaluation(output, inputs):
    """Reuse only a complete, hash-bound evaluation; legacy runs need an audit receipt."""
    if not (output / "metrics.json").is_file():
        return None
    metrics = json.loads((output / "metrics.json").read_text())
    if metrics.get("evaluation_inputs") != inputs:
        return None
    for split in ("valid", "test"):
        path = output / f"{split}_predictions.jsonl"
        if not path.is_file() or metrics.get("prediction_sha256", {}).get(split) != sha256_file(path):
            return None
    return metrics


def finalize(args):
    """Choose auxiliary lambda on ensemble validation BCE, then evaluate only winners."""
    torch.set_num_threads(args.threads)
    encoder_name = getattr(args, "encoder", "minimol")
    frozen = getattr(args, "frozen", False)
    data, graphs = load_prepared(args.prepared, with_graphs=False, encoder=encoder_name)
    labels = data["splits"]["valid"]["labels"].numpy()
    candidates, runs, signatures = {}, {}, set()
    prepared_sha256 = sha256_file(args.prepared / "manifest.json")
    indirect_only = getattr(args, "indirect_only", False)
    for weight in ([] if indirect_only else [0.0]) + ([] if frozen else args.aux_weights):
        root = args.output / f"aux_{weight:g}"
        binding = (dict(features_sha256=sha256_file(args.feature_cache / "receipt.json"),
                        encoder=encoder_name, frozen=frozen) if encoder_name != "minimol" else {})
        members = [validated_member(root / f"seed_{s}", prepared_sha256=prepared_sha256,
                                    seed=s, weight=weight, labels=labels, **binding) for s in args.seeds]
        runs[weight] = members
        for member in members:
            signatures.add(tuple(member["config"][k] for k in
                                 ("hidden_dim", "depth", "lr", "encoder_lr", "epochs", "batch_size")))
        if weight:
            scores = np.mean([member["scores"] for member in members], axis=0)
            candidates[weight] = ensemble_bce(labels, scores)
    if len(signatures) != 1:
        raise ValueError("Candidate runs differ in training parameters beyond seed/auxiliary weight")
    chosen = min(candidates, key=lambda weight: (candidates[weight], weight)) if candidates else 0.0
    selection = dict(criterion="minimum ensemble validation BCE", candidates=candidates,
                     seeds=args.seeds, selected_aux_weight=chosen, test_used=False)
    write_json_atomic(args.output / "selection.json", selection)
    evaluations = ([("frozen", 0.0)] if frozen else
                   ([] if indirect_only else [("finetuned", 0.0)]) + [("indirect", chosen)])
    for name, weight in evaluations:
        members = runs[weight]
        paths = [m["path"] for m in members]
        valid = np.mean([m["scores"] for m in members], axis=0)
        threshold, _ = choose_threshold(labels.tolist(), valid, "valid_macro_f1")
        out = args.output / name
        inputs = evaluation_inputs(args.prepared, members, threshold,
                                   feature_cache=args.feature_cache if encoder_name != "minimol" else None)
        cached = reusable_evaluation(out, inputs)
        if cached is not None:
            cached["selection"] = selection
            write_json_atomic(out / "metrics.json", cached)
            print(f"Reused verified evaluation: {out}", flush=True)
            continue
        if graphs is None:
            graphs = torch.load(args.prepared / "graphs.pt", weights_only=False)
        test_loader = loader(graphs, data["splits"]["test"], args.eval_batch_size, workers=args.workers)
        test_scores = []
        for path in paths:
            checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
            config = checkpoint["args"]
            encoder = None
            if encoder_name != "minimol":
                from predict.baselines.foundation_models import FoundationEncoder
                encoder = FoundationEncoder(encoder_name, args.runtime_root, args.feature_cache,
                                            data["smiles"], frozen=frozen)
            model = MultiTaskModel(len(data["conditions"]), len(data["buckets"]) if weight else 0,
                                   config["hidden_dim"], config["depth"], encoder=encoder)
            model.load_state_dict(checkpoint["state_dict"])
            model.precision = config.get("precision", "fp32")
            model.to(args.device)
            test_scores.append(predict(model, test_loader, args.device)[0])
            del model, checkpoint
        test = np.mean(test_scores, axis=0)
        out.mkdir(exist_ok=True)
        for split, scores in [("valid", valid), ("test", test)]:
            rows = data["splits"][split]
            with (out / f"{split}_predictions.jsonl").open("w") as handle:
                for i, score in enumerate(scores):
                    condition = (rows["condition_groups"][i] if "condition_groups" in rows else
                                 data["conditions"][int(rows["conditions"][i].argmax())])
                    handle.write(json.dumps(dict(index=i, drug=data["smiles"][int(rows["indices"][i])],
                                                 condition_group=condition,
                                                 label=int(rows["labels"][i]), probability=float(score),
                                                 prediction=int(score >= threshold))) + "\n")
        metrics = dict(name=name, aux_weight=weight, threshold=threshold,
                       valid=evaluate_metrics(labels.tolist(), valid, threshold),
                       test=evaluate_metrics(data["splits"]["test"]["labels"].tolist(), test, threshold),
                       checkpoints=[str(p / "best.pt") for p in paths], selection=selection,
                       evaluation_inputs=inputs,
                       prediction_sha256={s: sha256_file(out / f"{s}_predictions.jsonl") for s in ("valid", "test")})
        write_json_atomic(out / "metrics.json", metrics)
        print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "embed", "train", "finalize"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path, help="prepare: directory containing train/valid/test.jsonl")
    parser.add_argument("--auxiliary", type=Path, help="prepare: Evidence parent-pair-bucket Parquet")
    parser.add_argument("--prepare-workers", type=int, default=64)
    parser.add_argument("--condition-field", default="condition_group",
                        type=lambda value: None if value == "none" else value,
                        help="prepare: train-derived condition column, or 'none' for molecule-only heads")
    parser.add_argument("--preserve-main-split", action="store_true",
                        help="prepare: retain an external main split's existing parent overlap; auxiliary exclusion stays strict")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--aux-weight", type=float, default=0.0)
    parser.add_argument("--aux-weights", type=float, nargs="+", default=[0.1, 0.6, 1.0])
    parser.add_argument("--indirect-only", action="store_true",
                        help="finalize: evaluate only auxiliary candidates when refreshing auxiliary data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="finalize: ensemble members")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder", default="minimol")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.encoder != "minimol" and (args.runtime_root is None or args.feature_cache is None):
        parser.error("Foundation encoders require --runtime-root and --feature-cache")
    if args.frozen and (args.encoder == "minimol" or args.aux_weight):
        parser.error("--frozen requires a foundation encoder and zero auxiliary weight")
    if args.phase == "prepare" and (args.data_dir is None or args.auxiliary is None):
        parser.error("prepare requires --data-dir and --auxiliary")
    if args.phase != "prepare" and args.output is None:
        parser.error(f"{args.phase} requires --output")
    if args.epochs < 1 or args.batch_size < 2 or args.eval_batch_size < 1 or args.prepare_workers < 1:
        parser.error("epochs/workers/eval batch must be positive; training batch must be at least two")
    if not math.isfinite(args.aux_weight) or args.aux_weight < 0:
        parser.error("--aux-weight must be finite and nonnegative")
    if any(not math.isfinite(w) or w <= 0 for w in args.aux_weights) or len(set(args.aux_weights)) != len(args.aux_weights):
        parser.error("--aux-weights must be distinct finite positive values")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct")
    {"prepare": prepare, "embed": embed, "train": train, "finalize": finalize}[args.phase](args)


if __name__ == "__main__":
    main()
