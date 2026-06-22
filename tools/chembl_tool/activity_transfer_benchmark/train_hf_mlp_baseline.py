"""Train a Qwen endpoint + RDKit molecule MLP baseline for HF assay transfer."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import lightning.pytorch as pl
from torch import nn
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("MPLCONFIGDIR", "/local/tmp/matplotlib")

from tools.chembl_tool.activity_transfer_benchmark.run_llm_benchmark import (
    write_jsonl,
)


DEFAULT_PREPROCESSED_DIR = (
    "outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/"
    "qwen3_embedding_rdkit_v1/preprocessed"
)
DEFAULT_OUT_ROOT = (
    "outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/"
    "qwen3_embedding_rdkit_v1/runs"
)

DEFAULT_ID_TO_LABEL = {0: "different", 1: "similar"}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or time.strftime("mlp_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    config = vars(args) | {"run_id": run_id, "out_dir": str(out_dir)}
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.matmul_precision)

    feature_store = FeatureStore(Path(args.preprocessed_dir))
    stats = DescriptorStats.from_feature_store(feature_store, out_dir / "descriptor_stats.npz")
    train_dataset = AssayTransferIndexDataset(feature_store, "train")
    val_dataset = AssayTransferIndexDataset(feature_store, "validation")
    collator = FeatureCollator(feature_store, stats)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.eval_num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.eval_num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.eval_num_workers > 0 else None,
        collate_fn=collator,
    )

    model = AssayTransferMLP(
        endpoint_dim=feature_store.endpoint_dim,
        pair_dim=feature_store.pair_dim,
        endpoint_hidden=args.endpoint_hidden,
        endpoint_out=args.endpoint_out,
        pair_hidden=args.pair_hidden,
        pair_out=args.pair_out,
        fusion_hidden=args.fusion_hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    logger = False
    if args.wandb_mode != "disabled":
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_id,
            save_dir=str(out_dir),
            mode=args.wandb_mode,
            config=config,
        )

    validation_callback = FullValidationCallback(
        feature_store=feature_store,
        collator=collator,
        out_dir=out_dir,
        batch_size=args.eval_batch_size,
        num_workers=args.eval_num_workers,
        pin_memory=args.pin_memory,
        every_n_validation=args.full_eval_every_n_validation,
    )
    callbacks = [
        validation_callback,
        ModelCheckpoint(
            dirpath=out_dir / "checkpoints",
            filename="last-{step}",
            save_last=True,
            save_top_k=0,
            every_n_train_steps=args.checkpoint_every_steps,
        ),
    ]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        max_steps=args.max_steps,
        val_check_interval=args.val_every_steps,
        check_val_every_n_epoch=None,
        log_every_n_steps=args.log_every_steps,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if trainer.is_global_zero:
        save_rank_zero_model_checkpoint(
            out_dir / "checkpoints" / "final.ckpt",
            model,
            trainer,
            {"global_step": int(trainer.global_step), "final": True},
        )
        test_metrics = None
        if "test" in feature_store.row_indices:
            test_metrics = evaluate_and_write_split(
                model,
                feature_store,
                collator,
                out_dir,
                split="test",
                prefix="final_test",
                batch_size=args.eval_batch_size,
                num_workers=args.eval_num_workers,
                pin_memory=args.pin_memory,
                device=model.device,
                global_step=int(trainer.global_step),
            )
            log_metrics_to_wandb(trainer, test_metrics, prefix="test/final", step=int(trainer.global_step))
            best_ckpt_path = out_dir / "checkpoints" / "best.ckpt"
            if best_ckpt_path.exists():
                checkpoint = torch.load(best_ckpt_path, map_location=model.device, weights_only=False)
                model.load_state_dict(checkpoint["state_dict"])
                best_step = int(checkpoint.get("global_step", trainer.global_step))
                test_metrics = evaluate_and_write_split(
                    model,
                    feature_store,
                    collator,
                    out_dir,
                    split="test",
                    prefix="best_test",
                    batch_size=args.eval_batch_size,
                    num_workers=args.eval_num_workers,
                    pin_memory=args.pin_memory,
                    device=model.device,
                    global_step=best_step,
                )
                log_metrics_to_wandb(trainer, test_metrics, prefix="test/best", step=int(trainer.global_step))
        manifest = build_manifest(args, run_id, out_dir, feature_store, validation_callback)
        if test_metrics is not None:
            manifest["test_metrics"] = {
                "macro_f1": test_metrics["llm"]["macro_f1"],
                "accuracy": test_metrics["llm"]["accuracy"],
                "balanced_accuracy": test_metrics["llm"]["balanced_accuracy"],
                "n": test_metrics["n"],
                "n_ok": test_metrics["n_ok"],
            }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-dir", default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", type=int, default=13)

    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--endpoint-hidden", type=int, default=1024)
    parser.add_argument("--endpoint-out", type=int, default=512)
    parser.add_argument("--pair-hidden", type=int, default=2048)
    parser.add_argument("--pair-out", type=int, default=1024)
    parser.add_argument("--fusion-hidden", nargs="+", type=int, default=[1024, 512, 128])
    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)

    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu", "auto"])
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--strategy", default="ddp")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--matmul-precision", default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--val-every-steps", type=int, default=250)
    parser.add_argument("--full-eval-every-n-validation", type=int, default=1)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1000)
    parser.add_argument("--log-every-steps", type=int, default=50)

    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project", default="txagent-assay-transfer-mlp")
    parser.add_argument("--wandb-entity", default="")
    return parser.parse_args(argv)


@dataclass
class DescriptorStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_feature_store(cls, store: "FeatureStore", path: Path) -> "DescriptorStats":
        if store.molecule_backend != "rdkit":
            return cls(mean=np.zeros((0,), dtype=np.float32), std=np.ones((0,), dtype=np.float32))
        if path.exists():
            data = np.load(path)
            return cls(mean=data["mean"].astype(np.float32), std=data["std"].astype(np.float32))
        path.parent.mkdir(parents=True, exist_ok=True)
        train = store.row_indices["train"]
        train_molecules = np.unique(np.concatenate([train["molecule_a_index"], train["molecule_b_index"]]))
        descriptors = store.descriptors[train_molecules].astype(np.float32, copy=False)
        mean = np.nanmean(descriptors, axis=0).astype(np.float32)
        std = np.nanstd(descriptors, axis=0).astype(np.float32)
        mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
        np.savez_compressed(path, mean=mean, std=std, n_train_molecules=len(train_molecules))
        return cls(mean=mean, std=std)


class FeatureStore:
    def __init__(self, preprocessed_dir: Path) -> None:
        self.preprocessed_dir = preprocessed_dir
        self.manifest = json.loads((preprocessed_dir / "manifest.json").read_text(encoding="utf-8"))
        self.endpoint_embeddings = np.load(preprocessed_dir / "endpoint_embeddings.npy", mmap_mode="r")
        molecule_npz = np.load(preprocessed_dir / "molecule_features.npz")
        molecule_manifest = self.manifest.get("molecule_features") or {}
        self.molecule_backend = str(molecule_manifest.get("backend") or infer_molecule_backend(molecule_npz))
        self.fingerprints = None
        self.descriptors = None
        self.molformer_embeddings = None
        if self.molecule_backend == "rdkit":
            self.fingerprints = molecule_npz["fingerprints"]
            self.descriptors = molecule_npz["descriptors"].astype(np.float32, copy=False)
        elif self.molecule_backend == "molformer":
            self.molformer_embeddings = molecule_npz["molformer_embeddings"]
        else:
            raise ValueError(f"Unsupported molecule feature backend: {self.molecule_backend}")
        self.row_indices = {
            split_path.stem: load_row_index_npz(split_path)
            for split_path in sorted((preprocessed_dir / "row_indices").glob("*.npz"))
        }
        self.clean_rows = {
            split: read_clean_rows(preprocessed_dir / "clean_splits" / f"{split}.jsonl")
            for split in self.row_indices
            if split != "train" and (preprocessed_dir / "clean_splits" / f"{split}.jsonl").exists()
        }
        self.validation_rows = self.clean_rows["validation"]
        label_mapping = self.manifest.get("label_mapping") or {}
        raw_id_to_label = label_mapping.get("id_to_label") or {}
        self.id_to_label = {
            int(key): str(value)
            for key, value in raw_id_to_label.items()
            if str(key).lstrip("-").isdigit()
        } or DEFAULT_ID_TO_LABEL
        self.negative_label = str(label_mapping.get("negative_label") or self.id_to_label.get(0, "different"))
        self.positive_label = str(label_mapping.get("positive_label") or self.id_to_label.get(1, "similar"))
        self.validation_bucket_majorities = validation_bucket_majorities(self.validation_rows)
        self.endpoint_dim = int(self.endpoint_embeddings.shape[1])
        if self.molecule_backend == "rdkit":
            assert self.fingerprints is not None
            assert self.descriptors is not None
            self.fingerprint_dim = int(self.fingerprints.shape[1])
            self.descriptor_dim = int(self.descriptors.shape[1])
            self.molecule_embedding_dim = 0
            self.pair_dim = self.fingerprint_dim * 3 + self.descriptor_dim * 3
        else:
            assert self.molformer_embeddings is not None
            self.fingerprint_dim = 0
            self.descriptor_dim = 0
            self.molecule_embedding_dim = int(self.molformer_embeddings.shape[1])
            self.pair_dim = self.molecule_embedding_dim * 3


def infer_molecule_backend(molecule_npz: Any) -> str:
    files = set(molecule_npz.files)
    if "molformer_embeddings" in files:
        return "molformer"
    return "rdkit"


def load_row_index_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            "endpoint_index": data["endpoint_index"].astype(np.int32, copy=False),
            "molecule_a_index": data["molecule_a_index"].astype(np.int32, copy=False),
            "molecule_b_index": data["molecule_b_index"].astype(np.int32, copy=False),
            "label": data["label"].astype(np.int8, copy=False),
            "source_index": data["source_index"].astype(np.int32, copy=False),
        }


class AssayTransferIndexDataset(Dataset[tuple[int, int, int, int, int]]):
    def __init__(self, store: FeatureStore, split: str) -> None:
        self.rows = store.row_indices[split]

    def __len__(self) -> int:
        return int(self.rows["label"].shape[0])

    def __getitem__(self, index: int) -> tuple[int, int, int, int, int]:
        return (
            int(self.rows["endpoint_index"][index]),
            int(self.rows["molecule_a_index"][index]),
            int(self.rows["molecule_b_index"][index]),
            int(self.rows["label"][index]),
            int(self.rows["source_index"][index]),
        )


class FeatureCollator:
    def __init__(self, store: FeatureStore, stats: DescriptorStats) -> None:
        self.store = store
        self.mean = stats.mean
        self.std = stats.std

    def __call__(self, batch: list[tuple[int, int, int, int, int]]) -> dict[str, torch.Tensor]:
        endpoint_idx, mol_a_idx, mol_b_idx, labels, source_idx = map(np.asarray, zip(*batch))
        endpoint = np.asarray(self.store.endpoint_embeddings[endpoint_idx], dtype=np.float32)

        if self.store.molecule_backend == "molformer":
            assert self.store.molformer_embeddings is not None
            emb_a = np.asarray(self.store.molformer_embeddings[mol_a_idx], dtype=np.float32)
            emb_b = np.asarray(self.store.molformer_embeddings[mol_b_idx], dtype=np.float32)
            emb_diff = np.abs(emb_a - emb_b).astype(np.float32, copy=False)
            pair = np.concatenate([emb_a, emb_b, emb_diff], axis=1).astype(np.float32, copy=False)
            return {
                "endpoint": torch.from_numpy(endpoint),
                "pair": torch.from_numpy(pair),
                "label": torch.as_tensor(labels, dtype=torch.float32),
                "source_index": torch.as_tensor(source_idx, dtype=torch.long),
            }

        assert self.store.fingerprints is not None
        assert self.store.descriptors is not None
        fp_a_u8 = self.store.fingerprints[mol_a_idx]
        fp_b_u8 = self.store.fingerprints[mol_b_idx]
        fp_a = fp_a_u8.astype(np.float32, copy=False)
        fp_b = fp_b_u8.astype(np.float32, copy=False)
        fp_xor = np.bitwise_xor(fp_a_u8, fp_b_u8).astype(np.float32, copy=False)

        desc_a = normalize_descriptors(self.store.descriptors[mol_a_idx], self.mean, self.std)
        desc_b = normalize_descriptors(self.store.descriptors[mol_b_idx], self.mean, self.std)
        desc_diff = np.abs(desc_a - desc_b).astype(np.float32, copy=False)
        pair = np.concatenate([fp_a, fp_b, fp_xor, desc_a, desc_b, desc_diff], axis=1).astype(np.float32, copy=False)
        return {
            "endpoint": torch.from_numpy(endpoint),
            "pair": torch.from_numpy(pair),
            "label": torch.as_tensor(labels, dtype=torch.float32),
            "source_index": torch.as_tensor(source_idx, dtype=torch.long),
        }


def normalize_descriptors(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = (values.astype(np.float32, copy=False) - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(normalized, -10.0, 10.0).astype(np.float32, copy=False)


class AssayTransferMLP(pl.LightningModule):
    def __init__(
        self,
        *,
        endpoint_dim: int,
        pair_dim: int,
        endpoint_hidden: int,
        endpoint_out: int,
        pair_hidden: int,
        pair_out: int,
        fusion_hidden: list[int],
        dropout: float,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        max_steps: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.endpoint_tower = nn.Sequential(
            mlp_block(endpoint_dim, endpoint_hidden, dropout),
            mlp_block(endpoint_hidden, endpoint_out, dropout),
        )
        self.pair_tower = nn.Sequential(
            mlp_block(pair_dim, pair_hidden, dropout),
            mlp_block(pair_hidden, pair_out, dropout),
        )
        fusion_layers: list[nn.Module] = []
        current_dim = endpoint_out + pair_out
        for hidden_dim in fusion_hidden:
            fusion_layers.append(mlp_block(current_dim, hidden_dim, dropout))
            current_dim = hidden_dim
        fusion_layers.append(nn.Linear(current_dim, 1))
        self.fusion = nn.Sequential(*fusion_layers)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def forward(self, endpoint: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        endpoint_repr = self.endpoint_tower(endpoint)
        pair_repr = self.pair_tower(pair)
        return self.fusion(torch.cat([endpoint_repr, pair_repr], dim=1)).squeeze(1)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self(batch["endpoint"], batch["pair"])
        loss = self.loss_fn(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self(batch["endpoint"], batch["pair"])
        loss = self.loss_fn(logits, batch["label"])
        preds = (torch.sigmoid(logits) >= 0.5).float()
        acc = (preds == batch["label"]).float().mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/accuracy_batch", acc, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def lr_lambda(step: int) -> float:
            if self.warmup_steps > 0 and step < self.warmup_steps:
                return max(1e-6, float(step + 1) / float(self.warmup_steps))
            progress = min(1.0, max(0.0, (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class FullValidationCallback(pl.Callback):
    def __init__(
        self,
        *,
        feature_store: FeatureStore,
        collator: FeatureCollator,
        out_dir: Path,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        every_n_validation: int,
    ) -> None:
        self.feature_store = feature_store
        self.collator = collator
        self.out_dir = out_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.every_n_validation = every_n_validation
        self.validation_count = 0
        self.best_macro_f1 = -1.0
        self.best_step = -1

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        self.validation_count += 1
        should_eval = self.every_n_validation > 0 and self.validation_count % self.every_n_validation == 0
        if not should_eval:
            return

        trainer.strategy.barrier("full_validation_start")
        if trainer.is_global_zero and should_eval:
            metrics, predictions = evaluate_validation(
                pl_module,
                self.feature_store,
                self.collator,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                device=pl_module.device,
            )
            metrics["global_step"] = int(trainer.global_step)
            metrics["wall_time"] = time.time()
            write_jsonl(self.out_dir / "predictions.jsonl", predictions)
            (self.out_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            write_report(self.out_dir / "report_zh.md", metrics)
            macro_f1 = float(metrics["llm"]["macro_f1"])
            trainer.callback_metrics["val_macro_f1"] = torch.tensor(macro_f1, device=pl_module.device)
            pl_module.log("val_macro_f1", macro_f1, prog_bar=True, rank_zero_only=True)
            log_metrics_to_wandb(trainer, metrics)
            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1
                self.best_step = int(trainer.global_step)
                save_rank_zero_model_checkpoint(
                    self.out_dir / "checkpoints" / "best.ckpt",
                    pl_module,
                    trainer,
                    metrics,
                )
                shutil.copyfile(self.out_dir / "metrics.json", self.out_dir / "best_metrics.json")
                shutil.copyfile(self.out_dir / "predictions.jsonl", self.out_dir / "best_predictions.jsonl")
                log_metrics_to_wandb(trainer, metrics, prefix="best/validation", step=int(trainer.global_step))
        trainer.strategy.barrier("full_validation_end")


def save_rank_zero_model_checkpoint(path: Path, pl_module: pl.LightningModule, trainer: Any, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in pl_module.state_dict().items()},
        "hyper_parameters": dict(pl_module.hparams),
        "global_step": int(trainer.global_step),
        "pytorch-lightning_version": pl.__version__,
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    store: FeatureStore,
    collator: FeatureCollator,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return evaluate_split(
        model,
        store,
        collator,
        split="validation",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device=device,
    )


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    store: FeatureStore,
    collator: FeatureCollator,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = AssayTransferIndexDataset(store, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collator,
    )
    model.eval()
    probabilities: dict[int, float] = {}
    for batch in loader:
        endpoint = batch["endpoint"].to(device, non_blocking=True)
        pair = batch["pair"].to(device, non_blocking=True)
        logits = model(endpoint, pair)
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        source_indices = batch["source_index"].numpy()
        for source_index, prob in zip(source_indices, probs):
            probabilities[int(source_index)] = float(prob)

    predictions = []
    rows = store.clean_rows[split]
    default_majority = majority_label_generic([r["label"] for r in store.validation_rows])
    for row in rows:
        source_index = int(row["source_index"])
        prob = probabilities[source_index]
        prediction = store.positive_label if prob >= 0.5 else store.negative_label
        label = row["label"]
        metadata = row.get("metadata") or {}
        tanimoto = parse_float(metadata.get("weighted_tanimoto"))
        bucket_key = str(metadata.get("similarity_bucket"))
        has_transfer_labels = {store.positive_label, store.negative_label} == {"similar", "different"}
        tanimoto_0_50_prediction = "similar" if has_transfer_labels and tanimoto is not None and tanimoto >= 0.50 else ""
        tanimoto_0_48_prediction = "similar" if has_transfer_labels and tanimoto is not None and tanimoto >= 0.48 else ""
        predictions.append(
            {
                "status": "ok",
                "prediction": prediction,
                "prediction_score": prob,
                "label": label,
                "correct": prediction == label,
                "input_record": {
                    "input_format": "hf_prompt_completion",
                    "hf_metadata": metadata,
                    "metadata": metadata,
                    "tanimoto": tanimoto,
                    "similarity_bucket": metadata.get("similarity_bucket"),
                    "hf_sample_id": metadata.get("sample_id"),
                    "hf_pair_id": metadata.get("pair_id"),
                },
                "baseline_tanimoto_0_50_prediction": tanimoto_0_50_prediction,
                "baseline_tanimoto_0_48_prediction": tanimoto_0_48_prediction,
                "baseline_mcs_0_70_prediction": "",
                "baseline_similarity_bucket_majority_prediction": store.validation_bucket_majorities.get(
                    bucket_key,
                    default_majority,
                ),
                "baseline_assay_type_tanimoto_0_50_prediction": tanimoto_0_50_prediction,
                "usage": {},
                "tool_count": 0,
            }
        )
    metrics = compute_binary_metrics(predictions, positive_label=store.positive_label, negative_label=store.negative_label)
    return metrics, predictions


def evaluate_and_write_split(
    model: nn.Module,
    store: FeatureStore,
    collator: FeatureCollator,
    out_dir: Path,
    *,
    split: str,
    prefix: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
    global_step: int,
) -> dict[str, Any]:
    metrics, predictions = evaluate_split(
        model,
        store,
        collator,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device=device,
    )
    metrics["global_step"] = global_step
    metrics["wall_time"] = time.time()
    write_jsonl(out_dir / f"{prefix}_predictions.jsonl", predictions)
    (out_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir / f"{prefix}_report_zh.md", metrics)
    return metrics


def read_clean_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validation_bucket_majorities(rows: list[dict[str, Any]]) -> dict[str, str]:
    default = majority_label_generic(row["label"] for row in rows)
    counts: dict[str, Counter] = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        bucket = str(metadata.get("similarity_bucket"))
        counts.setdefault(bucket, Counter())[row["label"]] += 1
    return {bucket: majority_label_generic(counter.elements(), default_label=default) for bucket, counter in counts.items()}


def majority_label_generic(labels: Any, default_label: str = "") -> str:
    counter = Counter(labels)
    if not counter:
        return default_label
    return str(counter.most_common(1)[0][0])


def compute_binary_metrics(
    results: list[dict[str, Any]],
    *,
    positive_label: str,
    negative_label: str,
) -> dict[str, Any]:
    ok_results = [row for row in results if row.get("status") == "ok"]
    baseline_keys = {
        "tanimoto_0_50": "baseline_tanimoto_0_50_prediction",
        "tanimoto_0_48": "baseline_tanimoto_0_48_prediction",
        "mcs_0_70": "baseline_mcs_0_70_prediction",
        "similarity_bucket_majority": "baseline_similarity_bucket_majority_prediction",
        "assay_type_tanimoto_0_50": "baseline_assay_type_tanimoto_0_50_prediction",
    }
    metrics = {
        "n": len(results),
        "n_ok": len(ok_results),
        "n_failed": len(results) - len(ok_results),
        "positive_label": positive_label,
        "negative_label": negative_label,
        "llm": binary_metrics_for_predictions(ok_results, "prediction", positive_label, negative_label),
        "baselines": {
            name: binary_metrics_for_predictions(ok_results, key, positive_label, negative_label)
            for name, key in baseline_keys.items()
            if any(row.get(key) for row in ok_results)
        },
        "subsets": {},
        "usage": {},
        "tool_calls": 0,
    }
    add_binary_group_metrics(metrics, ok_results, baseline_keys, positive_label, negative_label)
    return metrics


def add_binary_group_metrics(
    metrics: dict[str, Any],
    ok_results: list[dict[str, Any]],
    baseline_keys: dict[str, str],
    positive_label: str,
    negative_label: str,
) -> None:
    group_specs = {
        "per_assay_type": lambda row: ((row.get("input_record") or {}).get("hf_metadata") or {}).get("assay_type"),
        "per_similarity_bucket": lambda row: str(((row.get("input_record") or {}).get("hf_metadata") or {}).get("similarity_bucket")),
        "per_eval_subset": lambda row: ((row.get("input_record") or {}).get("hf_metadata") or {}).get("eval_subset"),
    }
    for output_key, key_fn in group_specs.items():
        groups = sorted({str(key_fn(row)) for row in ok_results if key_fn(row) not in (None, "", "None")})
        if not groups:
            continue
        metrics[output_key] = {}
        for group in groups:
            rows = [row for row in ok_results if str(key_fn(row)) == group]
            metrics[output_key][group] = {
                "n": len(rows),
                "llm": binary_metrics_for_predictions(rows, "prediction", positive_label, negative_label),
                "baselines": {
                    name: binary_metrics_for_predictions(rows, key, positive_label, negative_label)
                    for name, key in baseline_keys.items()
                    if any(row.get(key) for row in rows)
                },
                "label_counts": dict(Counter(row.get("label") for row in rows)),
            }


def binary_metrics_for_predictions(
    rows: list[dict[str, Any]],
    prediction_key: str,
    positive_label: str,
    negative_label: str,
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = row.get(prediction_key)
        label = row.get("label")
        if pred not in {positive_label, negative_label} or label not in {positive_label, negative_label}:
            continue
        pred_positive = pred == positive_label
        true_positive = label == positive_label
        if pred_positive and true_positive:
            tp += 1
        elif pred_positive and not true_positive:
            fp += 1
        elif not pred_positive and true_positive:
            fn += 1
        else:
            tn += 1
    precision_positive = safe_div(tp, tp + fp)
    recall_positive = safe_div(tp, tp + fn)
    precision_negative = safe_div(tn, tn + fn)
    recall_negative = safe_div(tn, tn + fp)
    f1_positive = safe_div(2 * precision_positive * recall_positive, precision_positive + recall_positive)
    f1_negative = safe_div(2 * precision_negative * recall_negative, precision_negative + recall_negative)
    total = tp + fp + tn + fn
    metrics = {
        "n": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_div(tp + tn, total),
        "balanced_accuracy": (recall_positive + recall_negative) / 2.0,
        "macro_f1": (f1_positive + f1_negative) / 2.0,
        "precision_positive": precision_positive,
        "recall_positive": recall_positive,
        "precision_negative": precision_negative,
        "recall_negative": recall_negative,
        f"precision_{sanitize_metric_name(positive_label)}": precision_positive,
        f"recall_{sanitize_metric_name(positive_label)}": recall_positive,
        f"precision_{sanitize_metric_name(negative_label)}": precision_negative,
        f"recall_{sanitize_metric_name(negative_label)}": recall_negative,
    }
    if positive_label == "similar" and negative_label == "different":
        metrics.update(
            {
                "precision_similar": precision_positive,
                "recall_similar": recall_positive,
                "precision_different": precision_negative,
                "recall_different": recall_negative,
            }
        )
    else:
        metrics.update(
            {
                "precision_similar": precision_positive,
                "recall_similar": recall_positive,
                "precision_different": precision_negative,
                "recall_different": recall_negative,
            }
        )
    return metrics


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def parse_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def log_metrics_to_wandb(
    trainer: Any,
    metrics: dict[str, Any],
    *,
    prefix: str = "validation",
    step: int | None = None,
) -> None:
    logger = getattr(trainer, "logger", None)
    if not logger:
        return
    payload = flatten_metrics(metrics, prefix=prefix)
    logger.log_metrics(payload, step=int(step if step is not None else metrics.get("global_step", trainer.global_step)))


def flatten_metrics(metrics: dict[str, Any], *, prefix: str = "validation") -> dict[str, float]:
    positive_label = sanitize_metric_name(str(metrics.get("positive_label") or "positive"))
    negative_label = sanitize_metric_name(str(metrics.get("negative_label") or "negative"))
    payload = {
        f"{prefix}/accuracy": float(metrics["llm"]["accuracy"]),
        f"{prefix}/balanced_accuracy": float(metrics["llm"]["balanced_accuracy"]),
        f"{prefix}/macro_f1": float(metrics["llm"]["macro_f1"]),
        f"{prefix}/recall_{positive_label}": float(metrics["llm"]["recall_positive"]),
        f"{prefix}/recall_{negative_label}": float(metrics["llm"]["recall_negative"]),
    }
    for scope in ("per_similarity_bucket", "per_assay_type", "per_eval_subset"):
        for group, group_metrics in metrics.get(scope, {}).items():
            safe_group = sanitize_metric_name(str(group))
            llm = group_metrics.get("llm", {})
            group_prefix = f"{prefix}/{scope.replace('per_', '')}/{safe_group}"
            payload[f"{group_prefix}/macro_f1"] = float(llm.get("macro_f1", 0.0))
            payload[f"{group_prefix}/recall_{positive_label}"] = float(llm.get("recall_positive", 0.0))
            payload[f"{group_prefix}/recall_{negative_label}"] = float(llm.get("recall_negative", 0.0))
    return payload


def sanitize_metric_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    positive_label = str(metrics.get("positive_label") or "positive")
    negative_label = str(metrics.get("negative_label") or "negative")
    lines = [
        "# MLP Assay Transfer Report",
        "",
        f"- global_step: {metrics.get('global_step', '')}",
        f"- n: {metrics['n_ok']:,}/{metrics['n']:,}",
        f"- accuracy: {metrics['llm']['accuracy']:.4f}",
        f"- balanced accuracy: {metrics['llm']['balanced_accuracy']:.4f}",
        f"- macro-F1: {metrics['llm']['macro_f1']:.4f}",
        f"- recall {positive_label}: {metrics['llm']['recall_positive']:.4f}",
        f"- recall {negative_label}: {metrics['llm']['recall_negative']:.4f}",
        "",
    ]
    for scope in ("per_similarity_bucket", "per_assay_type", "per_eval_subset"):
        if scope not in metrics:
            continue
        lines.extend(
            [
                f"## {scope}",
                "",
                f"| group | n | macro-F1 | recall {positive_label} | recall {negative_label} |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for group, group_metrics in sorted(metrics[scope].items(), key=lambda item: str(item[0])):
            llm = group_metrics["llm"]
            lines.append(
                f"| {group} | {group_metrics['n']:,} | {llm['macro_f1']:.4f} | "
                f"{llm['recall_positive']:.4f} | {llm['recall_negative']:.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    args: argparse.Namespace,
    run_id: str,
    out_dir: Path,
    store: FeatureStore,
    callback: FullValidationCallback,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "preprocessed_dir": args.preprocessed_dir,
        "best_macro_f1": callback.best_macro_f1,
        "best_step": callback.best_step,
        "feature_shapes": {
            "endpoint_embeddings": list(store.endpoint_embeddings.shape),
            "molecule_backend": store.molecule_backend,
            "fingerprints": list(store.fingerprints.shape) if store.fingerprints is not None else [],
            "descriptors": list(store.descriptors.shape) if store.descriptors is not None else [],
            "molformer_embeddings": list(store.molformer_embeddings.shape)
            if store.molformer_embeddings is not None
            else [],
            "train_rows": int(store.row_indices["train"]["label"].shape[0]),
            "validation_rows": int(store.row_indices["validation"]["label"].shape[0]),
            "test_rows": int(store.row_indices["test"]["label"].shape[0]) if "test" in store.row_indices else 0,
        },
        "model": {
            "endpoint_hidden": args.endpoint_hidden,
            "endpoint_out": args.endpoint_out,
            "pair_hidden": args.pair_hidden,
            "pair_out": args.pair_out,
            "fusion_hidden": args.fusion_hidden,
            "dropout": args.dropout,
        },
        "training": vars(args),
    }


if __name__ == "__main__":
    raise SystemExit(main())
