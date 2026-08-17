"""Reproduce Starling paper Table 2 with the repository's MiniMol runtime.

The paper evaluates a TDC-only training pool and the same pool augmented with
literature-derived molecules on both TDC and literature test sets. Classification
training defaults to the released soft molecule labels; evaluation always uses
the released/derived hard labels. LD50 remains a continuous regression task.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch.utils.data import DataLoader

from baselines.minimol.embedding_runtime import (
    DEFAULT_MINIMOL_SOURCE,
    checkpoint_provenance,
    create_featurizer,
    embed_smiles,
)
from baselines.minimol.head_runtime import (
    EmbeddingDataset,
    evaluate_loss,
    make_model,
    train_one_epoch,
)
from baselines.minimol.starling_table2_data import (
    PAPER_TABLE2,
    TASKS,
    MoleculeTable,
    Table2Task,
    extraction_weights,
    file_receipt,
    filter_rdkit_invalid,
    load_task_tables,
)

DEFAULT_DATA_DIR = Path("data/starling_table2_v3/raw")
DEFAULT_OUTPUT_DIR = Path("outputs/baselines/minimol_starling_table2_v3")


@dataclass
class MemberReceipt:
    seed: int
    best_epoch: int
    best_valid_loss: float
    n_train: int
    n_valid: int
    n_train_scaffolds: int
    n_valid_scaffolds: int


def _parse_repetition_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("repetition indices must be comma-separated integers") from exc
    if not indices or len(indices) != len(set(indices)) or any(index < 1 for index in indices):
        raise argparse.ArgumentTypeError("repetition indices must be unique positive integers")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Optional shared all_molecules.pt cache; exact ordered-SMILES matching is required.",
    )
    parser.add_argument("--minimol-source", type=Path, default=DEFAULT_MINIMOL_SOURCE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--repetition-indices",
        type=_parse_repetition_indices,
        default=None,
        help="Comma-separated 1-based repetition subset for multi-GPU sharding.",
    )
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--train-label-policy",
        choices=("paper_soft", "hard"),
        default="paper_soft",
        help="paper_soft preserves released molecule-level targets; hard is a classification ablation.",
    )
    parser.add_argument(
        "--extraction-weight",
        choices=("none", "sqrt", "linear"),
        default="none",
        help="Optional molecule-row BCE weight derived from n_extractions; missing TDC counts use 1.",
    )
    parser.add_argument(
        "--evaluation-extraction-weight",
        choices=("none", "sqrt", "linear"),
        default="none",
        help="Optional n_extractions sample weight for classification AUROC; missing TDC counts use 1.",
    )
    parser.add_argument("--force-embed", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def selected_repetition_indices(args: argparse.Namespace) -> list[int]:
    indices = args.repetition_indices or list(range(1, args.repetitions + 1))
    if any(index > args.repetitions for index in indices):
        raise ValueError("--repetition-indices cannot exceed --repetitions")
    return sorted(indices)


def cantor_pairing(a: int, b: int) -> int:
    return (a + b) * (a + b + 1) // 2 + b


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scaffold_split_indices(
    smiles: list[str],
    *,
    seed: int,
    train_fraction: float = 0.875,
) -> tuple[list[int], list[int], dict[str, int]]:
    """Match TDC's seeded scaffold train/validation split for train_val pools."""
    RDLogger.DisableLog("rdApp.*")
    scaffold_to_indices: dict[str, set[int]] = {}
    for index, molecule in enumerate(smiles):
        mol = Chem.MolFromSmiles(molecule)
        if mol is None:
            raise ValueError(f"RDKit could not parse training SMILES at index {index}: {molecule}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffold_to_indices.setdefault(scaffold, set()).add(index)

    train_size = int(len(smiles) * train_fraction)
    valid_size = len(smiles) - train_size
    big: list[set[int]] = []
    small: list[set[int]] = []
    for index_set in scaffold_to_indices.values():
        (big if len(index_set) > valid_size / 2 else small).append(index_set)
    rng = random.Random(seed)
    rng.shuffle(big)
    rng.shuffle(small)

    train: list[int] = []
    valid: list[int] = []
    for index_set in big + small:
        destination = train if len(train) + len(index_set) <= train_size else valid
        destination.extend(index_set)

    train_scaffolds = {
        MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles[index]), includeChirality=False)
        for index in train
    }
    valid_scaffolds = {
        MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles[index]), includeChirality=False)
        for index in valid
    }
    if train_scaffolds & valid_scaffolds or sorted(train + valid) != list(range(len(smiles))):
        raise RuntimeError("scaffold split failed disjointness or coverage validation")
    return train, valid, {
        "n_train_scaffolds": len(train_scaffolds),
        "n_valid_scaffolds": len(valid_scaffolds),
    }


def _embedding_cache_path(output_dir: Path, args: argparse.Namespace) -> Path:
    return args.embedding_cache or output_dir / "embeddings" / "all_molecules.pt"


def materialize_embeddings(
    tables,
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    ordered_smiles = list(
        dict.fromkeys(
            tables.augmented_train.smiles
            + tables.tdc_test.smiles
            + tables.literature_test.smiles
        )
    )
    cache_path = _embedding_cache_path(output_dir, args)
    if cache_path.is_file() and not args.force_embed:
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("smiles") == ordered_smiles:
            tensor = payload["embeddings"].float()
            if tensor.shape == (len(ordered_smiles), 512) and torch.isfinite(tensor).all():
                return dict(zip(ordered_smiles, tensor, strict=True)), {
                    "cache_path": str(cache_path),
                    "cache_hit": True,
                    "n_molecules": len(ordered_smiles),
                }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    featurizer = create_featurizer(
        batch_size=args.embedding_batch_size,
        minimol_source=args.minimol_source,
        device=args.device,
    )
    tensor = embed_smiles(featurizer, ordered_smiles)
    torch.save({"smiles": ordered_smiles, "embeddings": tensor}, cache_path)
    return dict(zip(ordered_smiles, tensor, strict=True)), {
        "cache_path": str(cache_path),
        "cache_hit": False,
        "n_molecules": len(ordered_smiles),
    }


def table_embeddings(table: MoleculeTable, registry: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([registry[molecule] for molecule in table.smiles])


def predict_logits(model, embeddings: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(
        EmbeddingDataset(embeddings, np.zeros(len(embeddings))),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            chunks.append(model(inputs.to(device)).squeeze(-1).detach().cpu().numpy())
    return np.concatenate(chunks)


def metric_value(
    task: Table2Task,
    targets: list[float],
    scores: np.ndarray,
    sample_weights: list[float] | None = None,
) -> float:
    if task.metric == "auroc":
        return float(roc_auc_score(targets, scores, sample_weight=sample_weights))
    if sample_weights is not None:
        raise ValueError("weighted evaluation is only supported for classification AUROC")
    return float(mean_absolute_error(targets, scores))


def sample_weight_receipt(weights: list[float], policy: str) -> dict[str, float | str]:
    values = np.asarray(weights, dtype=float)
    return {
        "policy": policy,
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "sum": float(values.sum()),
    }


def train_condition(
    condition: str,
    train_table: MoleculeTable,
    evaluation_tables: dict[str, MoleculeTable],
    registry: dict[str, torch.Tensor],
    task: Table2Task,
    args: argparse.Namespace,
    output_dir: Path,
    repetition_indices: list[int],
) -> dict[str, object]:
    device = torch.device(args.device)
    train_embeddings = table_embeddings(train_table, registry)
    row_weights = extraction_weights(train_table, args.extraction_weight)
    evaluation_embeddings = {
        name: table_embeddings(table, registry) for name, table in evaluation_tables.items()
    }
    evaluation_weights = {
        name: extraction_weights(table, args.evaluation_extraction_weight)
        for name, table in evaluation_tables.items()
    }
    hparams = SimpleNamespace(
        hidden_dim=task.hidden_dim,
        depth=task.depth,
        dropout=args.dropout,
        lr=task.learning_rate,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        epochs=args.epochs,
    )
    repetitions: list[dict[str, object]] = []
    prediction_accumulator = {name: [] for name in evaluation_tables}

    for repetition in repetition_indices:
        models = []
        member_receipts: list[MemberReceipt] = []
        for member in range(1, args.ensemble_size + 1):
            seed = cantor_pairing(repetition, args.repetitions + member)
            set_seed(seed)
            train_indices, valid_indices, scaffold_counts = scaffold_split_indices(
                train_table.smiles,
                seed=seed,
            )
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                EmbeddingDataset(
                    train_embeddings[train_indices],
                    [train_table.targets[index] for index in train_indices],
                    [row_weights[index] for index in train_indices],
                ),
                batch_size=args.train_batch_size,
                shuffle=True,
                generator=generator,
            )
            valid_loader = DataLoader(
                EmbeddingDataset(
                    train_embeddings[valid_indices],
                    [train_table.targets[index] for index in valid_indices],
                    [row_weights[index] for index in valid_indices],
                ),
                batch_size=args.eval_batch_size,
                shuffle=False,
            )
            model, optimizer, scheduler, loss_fn = make_model(
                hparams,
                device,
                task_type=task.task_type,
            )
            best_model = None
            best_epoch = -1
            best_valid_loss = float("inf")
            for epoch in range(1, args.epochs + 1):
                train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    loss_fn,
                    device,
                )
                valid_loss = evaluate_loss(
                    model,
                    valid_loader,
                    loss_fn,
                    device,
                )
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    best_model = deepcopy(model).cpu()
            if best_model is None:
                raise RuntimeError("training did not produce a checkpoint")
            models.append(best_model)
            member_receipts.append(
                MemberReceipt(
                    seed=seed,
                    best_epoch=best_epoch,
                    best_valid_loss=best_valid_loss,
                    n_train=len(train_indices),
                    n_valid=len(valid_indices),
                    **scaffold_counts,
                )
            )
            print(
                f"[table2] task={task.slug} condition={condition} rep={repetition}/{args.repetitions} "
                f"member={member}/{args.ensemble_size} best_epoch={best_epoch} "
                f"valid_loss={best_valid_loss:.6f}",
                flush=True,
            )

        rep_metrics: dict[str, float] = {}
        for name, table in evaluation_tables.items():
            member_logits = []
            for model in models:
                model.to(device)
                member_logits.append(
                    predict_logits(model, evaluation_embeddings[name], args.eval_batch_size, device)
                )
                model.cpu()
            mean_logits = np.mean(np.stack(member_logits), axis=0)
            scores = 1.0 / (1.0 + np.exp(-mean_logits)) if task.task_type == "classification" else mean_logits
            rep_metrics[name] = metric_value(
                task,
                table.targets,
                scores,
                evaluation_weights[name]
                if args.evaluation_extraction_weight != "none"
                else None,
            )
            prediction_accumulator[name].append(scores)
        repetitions.append(
            {
                "repetition": repetition,
                "members": [asdict(receipt) for receipt in member_receipts],
                "metrics": rep_metrics,
            }
        )
        if args.save_checkpoints:
            checkpoint_dir = output_dir / "checkpoints" / condition
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"states": [model.state_dict() for model in models], "members": repetitions[-1]["members"]},
                checkpoint_dir / f"repetition_{repetition}.pt",
            )

    summary = {}
    for name in evaluation_tables:
        values = [float(row["metrics"][name]) for row in repetitions]
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
        average_scores = np.mean(np.stack(prediction_accumulator[name]), axis=0)
        prediction_path = output_dir / f"{condition}_{name}_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for molecule, target, raw_label, score in zip(
                evaluation_tables[name].smiles,
                evaluation_tables[name].targets,
                evaluation_tables[name].raw_labels,
                average_scores,
                strict=True,
            ):
                handle.write(
                    json.dumps(
                        {
                            "canonical_smiles": molecule,
                            "target": target,
                            "raw_label": raw_label,
                            "score": float(score),
                        }
                    )
                    + "\n"
                )
    return {
        "summary": summary,
        "repetitions": repetitions,
        "sample_weight": sample_weight_receipt(row_weights, args.extraction_weight),
    }


def main() -> None:
    args = parse_args()
    task = TASKS[args.task]
    if task.task_type == "regression" and args.train_label_policy == "hard":
        raise ValueError("--train-label-policy hard is only valid for classification tasks")
    if task.task_type == "regression" and args.extraction_weight != "none":
        raise ValueError("--extraction-weight is only supported for BCE classification tasks")
    if task.task_type == "regression" and args.evaluation_extraction_weight != "none":
        raise ValueError("--evaluation-extraction-weight is only supported for classification AUROC")
    run_label = args.train_label_policy
    if args.extraction_weight != "none":
        run_label += f"_{args.extraction_weight}_extraction_weight"
    if args.evaluation_extraction_weight != "none":
        run_label += f"_{args.evaluation_extraction_weight}_evaluation_weight"
    output_dir = args.output_dir / task.slug / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    repetition_indices = selected_repetition_indices(args)
    tables = load_task_tables(
        args.data_dir,
        task,
        train_label_policy=args.train_label_policy,
    )
    raw_sizes = {
        "tdc_train": len(tables.tdc_train.smiles),
        "augmented_train": len(tables.augmented_train.smiles),
        "tdc_test": len(tables.tdc_test.smiles),
        "literature_test": len(tables.literature_test.smiles),
    }
    tables, exclusions = filter_rdkit_invalid(tables)
    if exclusions:
        print(
            f"[table2] excluded {len(exclusions)} unique molecule(s) rejected by RDKit: "
            + ", ".join(str(row["canonical_smiles"]) for row in exclusions),
            flush=True,
        )
    print(
        f"[table2] task={task.slug} device={args.device} "
        f"tdc_train={len(tables.tdc_train.smiles)} augmented_train={len(tables.augmented_train.smiles)} "
        f"tdc_test={len(tables.tdc_test.smiles)} lit_test={len(tables.literature_test.smiles)}",
        flush=True,
    )
    registry, embedding_receipt = materialize_embeddings(
        tables,
        output_dir=output_dir,
        args=args,
    )
    evaluation_tables = {"tdc_test": tables.tdc_test, "literature_test": tables.literature_test}
    conditions = {
        "tdc_only": train_condition(
            "tdc_only",
            tables.tdc_train,
            evaluation_tables,
            registry,
            task,
            args,
            output_dir,
            repetition_indices,
        ),
        "augmented": train_condition(
            "augmented",
            tables.augmented_train,
            evaluation_tables,
            registry,
            task,
            args,
            output_dir,
            repetition_indices,
        ),
    }
    paper = PAPER_TABLE2[task.slug]
    comparison = None
    if repetition_indices == list(range(1, args.repetitions + 1)):
        observed = {
            "tdc_test": conditions["tdc_only"]["summary"]["tdc_test"]["mean"],
            "tdc_test_augmented": conditions["augmented"]["summary"]["tdc_test"]["mean"],
            "literature_test": conditions["tdc_only"]["summary"]["literature_test"]["mean"],
            "literature_test_augmented": conditions["augmented"]["summary"]["literature_test"]["mean"],
        }
        comparison = {
            key: {
                "paper": paper[key],
                "observed": float(observed[key]),
                "observed_minus_paper": float(observed[key] - paper[key]),
            }
            for key in paper
        }
    result = {
        "type": "starling_table2_minimol_reproduction.v1",
        "task": asdict(task),
        "repetition_indices": repetition_indices,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {
            "files": {name: file_receipt(path) for name, path in tables.files.items()},
            "raw_sizes": raw_sizes,
            "sizes": {
                "tdc_train": len(tables.tdc_train.smiles),
                "augmented_train": len(tables.augmented_train.smiles),
                "tdc_test": len(tables.tdc_test.smiles),
                "literature_test": len(tables.literature_test.smiles),
            },
            "exclusions": exclusions,
            "evaluation_sample_weight": {
                name: sample_weight_receipt(
                    extraction_weights(table, args.evaluation_extraction_weight),
                    args.evaluation_extraction_weight,
                )
                for name, table in evaluation_tables.items()
            },
        },
        "minimol": checkpoint_provenance(args.minimol_source),
        "embeddings": embedding_receipt,
        "conditions": conditions,
        "paper_comparison": comparison,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if comparison is not None:
        print(json.dumps(comparison, indent=2), flush=True)
    print(f"[table2] wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
