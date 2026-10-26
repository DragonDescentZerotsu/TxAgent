"""Prepare, embed, train and evaluate parent-disjoint MiniMol auxiliary tasks."""

from __future__ import annotations

import argparse
import json
import math
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
    def __init__(self, conditions: int, buckets: int, hidden_dim: int, depth: int):
        super().__init__()
        self.encoder = MiniMolEncoder()
        self.head = TaskHead(input_dim=512 + conditions, hidden_dim=hidden_dim, depth=depth)
        self.auxiliary = nn.Linear(512, buckets) if buckets else None

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
        embeddings = model.encoder(graph.to(device, non_blocking=True))
        output = model.main_logits(embeddings, conditions.to(device, non_blocking=True))
        logits.append(output.float().cpu())
        labels.append(target)
    logits, labels = torch.cat(logits), torch.cat(labels)
    return logits.sigmoid().numpy(), float(nn.functional.binary_cross_entropy_with_logits(logits, labels))


def load_prepared(root, *, with_graphs=True):
    manifest = json.loads((root / "manifest.json").read_text())
    if sha256_file(root / "data.pt") != manifest["data_sha256"]:
        raise ValueError("Prepared target cache hash changed")
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
    data, graphs = load_prepared(args.prepared)
    device = torch.device(args.device)
    model = MultiTaskModel(len(data["conditions"]), len(data["buckets"]) if args.aux_weight else 0,
                           args.hidden_dim, args.depth).to(device)
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
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        totals = np.zeros(2)
        aux_iterator = iter(auxiliary_loader) if auxiliary_loader is not None else None
        for graph, conditions, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            graph = graph.to(device, non_blocking=True)
            main_embeddings = model.encoder(graph)
            logits = model.main_logits(main_embeddings, conditions.to(device, non_blocking=True))
            main_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device, non_blocking=True))
            # Release main activations before auxiliary forward, keeping gradients accumulated.
            main_loss.backward()
            aux_loss = torch.zeros((), device=device)
            if aux_iterator is not None:
                aux_graph, buckets, targets, binary = next(aux_iterator)
                embeddings = model.encoder(aux_graph.to(device, non_blocking=True))
                aux_logits = model.auxiliary(embeddings).gather(1, buckets.to(device)[:, None]).squeeze(1)
                aux_loss = auxiliary_loss(aux_logits, targets.to(device), binary.to(device))
                (args.aux_weight * aux_loss).backward()
            if not torch.isfinite(main_loss + aux_loss):
                raise FloatingPointError("Nonfinite loss")
            if not gradient_receipt:
                nonzero = sum(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.encoder.parameters())
                if not nonzero:
                    raise RuntimeError("Encoder gradients are disconnected")
                gradient_receipt = dict(encoder_parameters_with_nonzero_gradient=nonzero)
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
                              prepared_sha256=sha256_file(args.prepared / "manifest.json"))
            np.save(args.output / "valid_scores.npy", scores)
            checkpoint["valid_scores_sha256"] = sha256_file(args.output / "valid_scores.npy")
            torch.save(checkpoint, args.output / "best.pt")
        row = dict(epoch=epoch, train_bce=float(totals[0] / len(train_loader)),
                   auxiliary_loss=float(totals[1] / len(train_loader)), valid_bce=valid_bce,
                   seconds=time.monotonic() - epoch_start, best_epoch=best_epoch)
        history.append(row)
        write_json_atomic(args.output / "progress.json", dict(history=history, best_epoch=best_epoch))
        print(json.dumps(row), flush=True)
    write_json_atomic(args.output / "training.json", dict(
        seed=args.seed, aux_weight=args.aux_weight, best_epoch=best_epoch, best_valid_bce=best_loss,
        history=history, gradient_check=gradient_receipt, elapsed_seconds=time.monotonic() - started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        checkpoint_sha256=sha256_file(args.output / "best.pt"),
        test_used=False,
    ))


def ensemble_bce(labels, scores):
    p = np.clip(scores.astype(np.float64), 1e-12, 1 - 1e-12)
    return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())


def validated_member(path, *, prepared_sha256, seed, weight, labels):
    """Check every candidate before validation selection, including non-winners."""
    receipt = json.loads((path / "training.json").read_text())
    digest = sha256_file(path / "best.pt")
    if digest != receipt["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint changed: {path}")
    checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["args"]
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
    # Legacy checkpoints precede the score hash; bind their scores to saved BCE.
    if not np.isclose(ensemble_bce(labels, scores), checkpoint["valid_bce"], rtol=1e-4, atol=1e-5):
        raise ValueError(f"Validation scores do not match checkpoint BCE: {path}")
    return dict(path=path, config=config, scores=scores, checkpoint_sha256=digest,
                valid_scores_sha256=score_hash)


def evaluation_inputs(prepared, members, threshold):
    return dict(prepared_sha256=sha256_file(prepared / "manifest.json"),
                graph_sha256=json.loads((prepared / "graphs.json").read_text())["graph_sha256"],
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
    data, graphs = load_prepared(args.prepared, with_graphs=False)
    labels = data["splits"]["valid"]["labels"].numpy()
    candidates, runs, signatures = {}, {}, set()
    prepared_sha256 = sha256_file(args.prepared / "manifest.json")
    indirect_only = getattr(args, "indirect_only", False)
    for weight in ([] if indirect_only else [0.0]) + args.aux_weights:
        root = args.output / f"aux_{weight:g}"
        members = [validated_member(root / f"seed_{s}", prepared_sha256=prepared_sha256,
                                    seed=s, weight=weight, labels=labels) for s in args.seeds]
        runs[weight] = members
        for member in members:
            signatures.add(tuple(member["config"][k] for k in
                                 ("hidden_dim", "depth", "lr", "encoder_lr", "epochs", "batch_size")))
        if weight:
            scores = np.mean([member["scores"] for member in members], axis=0)
            candidates[weight] = ensemble_bce(labels, scores)
    if len(signatures) != 1:
        raise ValueError("Candidate runs differ in training parameters beyond seed/auxiliary weight")
    chosen = min(candidates, key=lambda weight: (candidates[weight], weight))
    selection = dict(criterion="minimum ensemble validation BCE", candidates=candidates,
                     seeds=args.seeds, selected_aux_weight=chosen, test_used=False)
    write_json_atomic(args.output / "selection.json", selection)
    evaluations = ([] if indirect_only else [("finetuned", 0.0)]) + [("indirect", chosen)]
    for name, weight in evaluations:
        members = runs[weight]
        paths = [m["path"] for m in members]
        valid = np.mean([m["scores"] for m in members], axis=0)
        threshold, _ = choose_threshold(labels.tolist(), valid, "valid_macro_f1")
        out = args.output / name
        inputs = evaluation_inputs(args.prepared, members, threshold)
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
            model = MultiTaskModel(len(data["conditions"]), len(data["buckets"]) if weight else 0,
                                   config["hidden_dim"], config["depth"])
            model.load_state_dict(checkpoint["state_dict"])
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
    args = parser.parse_args()
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
