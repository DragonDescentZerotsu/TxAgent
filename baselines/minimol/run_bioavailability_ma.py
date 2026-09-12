"""Train and evaluate a MiniMol baseline on Bioavailability_Ma.

This follows the downstream head used by MiniMol's TDC leaderboard script, but
uses the repo's fixed train/valid/test JSONL splits instead of TDC folds.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from baselines.minimol.embedding_runtime import (
    DEFAULT_MINIMOL_SOURCE,
    create_featurizer,
    embed_smiles,
)
from baselines.minimol.condition_features import (
    append_condition_one_hot,
    condition_feature_contract,
    condition_vocabulary,
)
from baselines.minimol.head_runtime import (
    EmbeddingDataset,
    evaluate_loss,
    make_model,
    predict_scores,
    train_one_epoch,
)

DEFAULT_DATA_DIR = Path("data/processed/Bioavailability_Ma")
DEFAULT_OUTPUT_DIR = Path("outputs/baselines/minimol/bioavailability_ma")


@dataclass
class SplitData:
    smiles: list[str]
    labels: list[int]
    conditions: list[str] | None = None


@dataclass
class RunResult:
    seed: int
    best_epoch: int
    best_valid_loss: float | None
    valid_threshold: float | None
    valid_macro_f1: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unseen-condition-policy", choices=("error", "zero"), default="error",
                        help="Explicitly encode unseen evaluation conditions as all-zero features, or fail (default).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--reuse-embedding-cache-dir",
        type=Path,
        action="append",
        default=[],
        help="Reuse frozen molecule-only embeddings by exact SMILES match; may be repeated.",
    )
    parser.add_argument("--minimol-source", type=Path, default=DEFAULT_MINIMOL_SOURCE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3, choices=[3, 4])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold-strategy", choices=["fixed_0.5", "valid_macro_f1"], default="fixed_0.5")
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help=(
            "Optional train-only calibrated decision threshold. Supported with "
            "--train-all; probabilities and AUROC are unchanged."
        ),
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("valid", "test"),
        default="test",
        help="Split evaluated after training; valid is supported with --train-all only.",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train for all configured epochs on train.jsonl, without requiring or selecting on valid.jsonl.",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Materialize validated embedding caches and exit without fitting a task head.",
    )
    parser.add_argument("--force-embed", action="store_true", help="Ignore cached MiniMol embeddings.")
    parser.add_argument(
        "--condition-field",
        default=None,
        help=(
            "Optional categorical JSONL field appended to MiniMol embeddings as a "
            "train-derived one-hot feature."
        ),
    )
    return parser.parse_args()


def load_split(path: Path, *, condition_field: str | None = None) -> SplitData:
    smiles: list[str] = []
    labels: list[int] = []
    conditions: list[str] | None = [] if condition_field else None
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            row = json.loads(line)
            try:
                smiles.append(str(row["drug"]))
                labels.append(int(row["Y"]))
                if conditions is not None and condition_field is not None:
                    value = str(row[condition_field]).strip()
                    if not value:
                        raise ValueError(
                            f"{path}:{line_number} has an empty {condition_field!r}"
                        )
                    conditions.append(value)
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} is missing required key {exc!s}") from exc
    return SplitData(smiles=smiles, labels=labels, conditions=conditions)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cache_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / "embeddings" / f"{split_name}.pt"


def embedding_cache_path(args: argparse.Namespace, split_name: str) -> Path:
    if args.embedding_cache_dir is not None:
        return args.embedding_cache_dir / f"{split_name}.pt"
    return cache_path(args.output_dir, split_name)


def featurize_splits(
    splits: dict[str, SplitData],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    pending: dict[str, SplitData] = {}
    for split_name, split in splits.items():
        path = embedding_cache_path(args, split_name)
        if path.exists() and not args.force_embed:
            payload = torch.load(path, map_location="cpu")
            if payload["smiles"] == split.smiles:
                embeddings[split_name] = payload["embeddings"].float()
                continue
            print(f"[minimol] cache mismatch for {split_name}; recomputing embeddings")
        pending[split_name] = split

    if pending:
        reusable, reuse_files = _load_reusable_embeddings(args.reuse_embedding_cache_dir)
        missing_smiles = list(
            dict.fromkeys(
                smiles
                for split in pending.values()
                for smiles in split.smiles
                if smiles not in reusable
            )
        )
        if missing_smiles:
            featurizer = create_featurizer(
                batch_size=args.embedding_batch_size,
                minimol_source=args.minimol_source,
                device=args.device,
            )
            missing_embeddings = embed_smiles(featurizer, missing_smiles)
            reusable.update(zip(missing_smiles, missing_embeddings, strict=True))
        missing_set = set(missing_smiles)
        reuse_counts: dict[str, dict[str, int]] = {}
        for split_name, split in pending.items():
            n_reused = sum(smiles not in missing_set for smiles in split.smiles)
            print(
                f"[minimol] materializing {split_name}: {len(split.smiles)} molecules "
                f"({n_reused} reused, {len(split.smiles) - n_reused} newly embedded)"
            )
            tensor = torch.stack([reusable[smiles] for smiles in split.smiles]).float()
            path = embedding_cache_path(args, split_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"smiles": split.smiles, "labels": split.labels, "embeddings": tensor},
                path,
            )
            embeddings[split_name] = tensor
            reuse_counts[split_name] = {
                "n_rows": len(split.smiles),
                "n_reused": n_reused,
                "n_newly_embedded": len(split.smiles) - n_reused,
            }
        reuse_manifest = {
            "type": "minimol_embedding_cache_reuse.v1",
            "match_key": "exact_smiles",
            "label_fields_reused": False,
            "source_cache_files": [str(path) for path in reuse_files],
            "n_source_molecules": len(reusable) - len(missing_smiles),
            "n_unique_newly_embedded": len(missing_smiles),
            "splits": reuse_counts,
        }
        reuse_manifest_path = args.output_dir / "embedding_reuse_manifest.json"
        reuse_manifest_path.write_text(
            json.dumps(reuse_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    return embeddings


def _load_reusable_embeddings(
    cache_dirs: list[Path],
) -> tuple[dict[str, torch.Tensor], list[Path]]:
    registry: dict[str, torch.Tensor] = {}
    files = sorted(
        path
        for cache_dir in cache_dirs
        for path in cache_dir.glob("*.pt")
        if path.is_file()
    )
    for path in files:
        payload = torch.load(path, map_location="cpu")
        smiles = list(map(str, payload["smiles"]))
        tensor = payload["embeddings"].float()
        if len(smiles) != len(tensor):
            raise ValueError(f"Embedding cache row mismatch: {path}")
        for molecule, row in zip(smiles, tensor, strict=True):
            previous = registry.get(molecule)
            if previous is not None and not torch.allclose(previous, row, rtol=1e-5, atol=1e-6):
                raise ValueError(f"Conflicting cached MiniMol embedding for {molecule!r}")
            registry[molecule] = row
    return registry, files


def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    return predict_scores(model, loader, device, task_type="classification")


def choose_threshold(y_true: list[int], y_score: np.ndarray, strategy: str) -> tuple[float, float]:
    if strategy == "fixed_0.5":
        threshold = 0.5
        return threshold, float(f1_score(y_true, y_score >= threshold, average="macro"))

    candidates = sorted(set(float(x) for x in y_score))
    thresholds = [0.5]
    thresholds.extend(candidates)
    thresholds.extend((a + b) / 2 for a, b in zip(candidates, candidates[1:]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        score = float(f1_score(y_true, y_score >= threshold, average="macro"))
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_threshold = float(threshold)
            best_score = score
    return best_threshold, best_score


def evaluate_metrics(y_true: list[int], y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "threshold": float(threshold),
    }


def main() -> None:
    args = parse_args()
    if args.decision_threshold is not None:
        if not args.train_all:
            raise ValueError("--decision-threshold requires --train-all")
        if not 0.0 <= args.decision_threshold <= 1.0:
            raise ValueError("--decision-threshold must be between 0 and 1")
    if args.evaluation_split == "valid" and not args.train_all:
        raise ValueError("--evaluation-split valid requires --train-all to avoid validation selection leakage")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache_dir = args.output_dir / ".matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    device = torch.device(args.device)

    train = load_split(
        args.data_dir / "train.jsonl", condition_field=args.condition_field
    )
    valid = (
        None
        if args.train_all
        else load_split(
            args.data_dir / "valid.jsonl", condition_field=args.condition_field
        )
    )
    test = load_split(
        args.data_dir / f"{args.evaluation_split}.jsonl",
        condition_field=args.condition_field,
    )

    print(
        "[minimol] loaded splits: "
        f"train={len(train.labels)} "
        f"valid={len(valid.labels) if valid is not None else 'not_used'} "
        f"evaluation={args.evaluation_split}:{len(test.labels)} device={device}"
    )

    embedding_splits = {"train": train, args.evaluation_split: test}
    if valid is not None:
        embedding_splits["valid"] = valid
    split_embeddings = featurize_splits(embedding_splits, args)
    if args.embeddings_only:
        (args.output_dir / "embeddings_only_manifest.json").write_text(
            json.dumps(
                {
                    "type": "minimol_embeddings_only.v1",
                    "data_dir": str(args.data_dir),
                    "embedding_cache_dir": str(args.embedding_cache_dir),
                    "evaluation_split": args.evaluation_split,
                    "splits": {
                        name: {
                            "n_rows": len(embedding_splits[name].smiles),
                            "embedding_shape": list(tensor.shape),
                        }
                        for name, tensor in split_embeddings.items()
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("[minimol] embeddings-only cache materialization complete")
        return
    molecule_embedding_dim = int(split_embeddings["train"].shape[1])
    vocabulary = (
        condition_vocabulary(train.conditions)
        if train.conditions is not None
        else None
    )
    train_embeddings = append_condition_one_hot(
        split_embeddings["train"], train.conditions, vocabulary,
        unseen_policy=getattr(args, "unseen_condition_policy", "error")
    )
    valid_embeddings = (
        append_condition_one_hot(
            split_embeddings["valid"], valid.conditions, vocabulary,
            unseen_policy=getattr(args, "unseen_condition_policy", "error")
        )
        if valid is not None
        else None
    )
    test_embeddings = append_condition_one_hot(
        split_embeddings[args.evaluation_split], test.conditions, vocabulary,
        unseen_policy=getattr(args, "unseen_condition_policy", "error")
    )
    feature_contract = condition_feature_contract(
        field=args.condition_field,
        vocabulary=vocabulary,
        molecule_embedding_dim=molecule_embedding_dim,
        unseen_policy=getattr(args, "unseen_condition_policy", "error"),
    )

    valid_loader = (
        DataLoader(
            EmbeddingDataset(valid_embeddings, valid.labels),
            batch_size=args.eval_batch_size,
            shuffle=False,
        )
        if valid is not None and valid_embeddings is not None
        else None
    )
    test_loader = DataLoader(EmbeddingDataset(test_embeddings, test.labels), batch_size=args.eval_batch_size, shuffle=False)

    best_models: list[nn.Module] = []
    run_results: list[RunResult] = []

    for member_idx in range(args.ensemble_size):
        seed = args.seed + member_idx
        set_seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            EmbeddingDataset(train_embeddings, train.labels),
            batch_size=args.train_batch_size,
            shuffle=True,
            generator=generator,
        )
        model, optimizer, scheduler, loss_fn = make_model(
            args, device, input_dim=train_embeddings.shape[1]
        )
        best_epoch = -1
        best_valid_loss = float("inf")
        best_model = None

        for epoch in range(args.epochs):
            train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
            if valid_loader is None:
                print(
                    f"[minimol] member={member_idx + 1}/{args.ensemble_size} "
                    f"epoch={epoch + 1}/{args.epochs} train_all=true"
                )
            else:
                valid_loss = evaluate_loss(model, valid_loader, loss_fn, device)
                if valid_loss < best_valid_loss:
                    best_epoch = epoch + 1
                    best_valid_loss = valid_loss
                    best_model = deepcopy(model).cpu()
                print(
                    f"[minimol] member={member_idx + 1}/{args.ensemble_size} "
                    f"epoch={epoch + 1}/{args.epochs} valid_loss={valid_loss:.6f} "
                    f"best_epoch={best_epoch}"
                )

        if valid_loader is None:
            best_epoch = args.epochs
            best_model = deepcopy(model).cpu()

        if best_model is None:
            raise RuntimeError("No model was trained")

        if valid_loader is not None and valid is not None:
            best_model.to(device)
            valid_score = predict_proba(best_model, valid_loader, device)
            threshold, valid_macro_f1 = choose_threshold(valid.labels, valid_score, args.threshold_strategy)
            best_model.cpu()
        else:
            threshold = 0.5
            valid_macro_f1 = None
        best_models.append(best_model)
        run_results.append(
            RunResult(
                seed=seed,
                best_epoch=best_epoch,
                best_valid_loss=float(best_valid_loss) if valid_loader is not None else None,
                valid_threshold=threshold,
                valid_macro_f1=valid_macro_f1,
            )
        )

    valid_member_scores = []
    test_member_scores = []
    for model in best_models:
        model.to(device)
        if valid_loader is not None:
            valid_member_scores.append(predict_proba(model, valid_loader, device))
        test_member_scores.append(predict_proba(model, test_loader, device))
        model.cpu()

    test_scores = np.mean(np.stack(test_member_scores), axis=0)
    if valid is not None:
        valid_scores = np.mean(np.stack(valid_member_scores), axis=0)
        threshold, valid_macro_f1 = choose_threshold(valid.labels, valid_scores, args.threshold_strategy)
        valid_metrics = evaluate_metrics(valid.labels, valid_scores, threshold)
        valid_metrics_fixed = evaluate_metrics(valid.labels, valid_scores, 0.5)
        valid_tuned_threshold, _ = choose_threshold(valid.labels, valid_scores, "valid_macro_f1")
        valid_metrics_tuned = evaluate_metrics(valid.labels, valid_scores, valid_tuned_threshold)
        test_metrics_tuned = evaluate_metrics(test.labels, test_scores, valid_tuned_threshold)
    else:
        threshold = (
            float(args.decision_threshold)
            if args.decision_threshold is not None
            else 0.5
        )
        valid_macro_f1 = None
        valid_metrics = None
        valid_metrics_fixed = None
        valid_metrics_tuned = None
        test_metrics_tuned = None
    test_metrics = evaluate_metrics(test.labels, test_scores, threshold)
    test_metrics_fixed = evaluate_metrics(test.labels, test_scores, 0.5)

    split_sizes = {
        "train": len(train.labels),
        "valid": len(valid.labels) if valid is not None else 0,
        "test": 0,
        args.evaluation_split: len(test.labels),
        "evaluation": len(test.labels),
    }
    output = {
        "args": {key: _json_safe(value) for key, value in vars(args).items()},
        "splits": split_sizes,
        "evaluation_split": args.evaluation_split,
        "condition_features": feature_contract,
        "embedding_reuse_manifest": (
            str(args.output_dir / "embedding_reuse_manifest.json")
            if (args.output_dir / "embedding_reuse_manifest.json").exists()
            else None
        ),
        "model_selection": {
            "criterion": (
                "fixed configured epochs using all training molecules"
                if args.train_all
                else "lowest validation BCE loss per ensemble member"
            ),
            "threshold_strategy": (
                "configured_train_only_oof"
                if args.decision_threshold is not None
                else "fixed_0.5"
                if args.train_all
                else args.threshold_strategy
            ),
            "decision_threshold_source": (
                "configured_train_only_oof"
                if args.decision_threshold is not None
                else "fixed_0.5"
                if args.train_all
                else args.threshold_strategy
            ),
            "ensemble_valid_macro_f1_at_threshold": (
                float(valid_macro_f1) if valid_macro_f1 is not None else None
            ),
        },
        "members": [asdict(result) for result in run_results],
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "valid_metrics_fixed_0.5": valid_metrics_fixed,
        "test_metrics_fixed_0.5": test_metrics_fixed,
        "evaluation_metrics": test_metrics,
        "evaluation_metrics_fixed_0.5": test_metrics_fixed,
        "valid_metrics_valid_macro_f1_threshold": valid_metrics_tuned,
        "test_metrics_valid_macro_f1_threshold": test_metrics_tuned,
    }

    metrics_path = args.output_dir / "metrics.json"
    predictions_path = args.output_dir / f"{args.evaluation_split}_predictions.jsonl"
    checkpoint_path = args.output_dir / "best_ensemble.pt"
    with metrics_path.open("w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    with predictions_path.open("w") as f:
        for index, (smiles, label, score) in enumerate(
            zip(test.smiles, test.labels, test_scores, strict=True)
        ):
            row = {"drug": smiles, "Y": label, "score": float(score), "prediction": int(score >= threshold)}
            if args.condition_field is not None and test.conditions is not None:
                row[args.condition_field] = test.conditions[index]
            f.write(json.dumps(row) + "\n")
    torch.save(
        {
            "model_state_dicts": [model.state_dict() for model in best_models],
            "run_results": [asdict(result) for result in run_results],
            "threshold": threshold,
            "hparams": output["args"],
            "condition_features": feature_contract,
        },
        checkpoint_path,
    )

    print("[minimol] valid metrics:", json.dumps(valid_metrics, sort_keys=True))
    print(
        f"[minimol] {args.evaluation_split} metrics:",
        json.dumps(test_metrics, sort_keys=True),
    )
    print(f"[minimol] wrote {metrics_path}")
    print(f"[minimol] wrote {predictions_path}")
    print(f"[minimol] wrote {checkpoint_path}")


if __name__ == "__main__":
    main()
