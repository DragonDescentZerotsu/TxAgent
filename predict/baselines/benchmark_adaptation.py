"""Matched training-step timings; diagnostic only, never a paper evaluation."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

from predict.baselines.foundation_models import FoundationEncoder, MODELS
from predict.baselines.minimol.run_multitask import (
    MultiTaskModel, auxiliary_loss, loader, load_prepared, set_seed,
)
from predict.utils.json import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--mode", choices=("full_fp32", "full_bf16", "lora_bf16"), required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--unimol-qv-only", action="store_true")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve completed measurements; choose a new output")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    prepared = args.run / "inputs/conditioned_train/bbb_martins"
    cache = args.run / "features" / args.model / "conditioned_train/bbb_martins"
    data, graphs = load_prepared(prepared, encoder=args.model)
    encoder = FoundationEncoder(args.model, args.runtime_root, cache, data["smiles"])
    if args.no_activation_checkpointing:
        if args.model != "unimol":
            raise ValueError("Checkpoint toggle applies only to Uni-Mol")
        encoder.network.encoder.layers = nn.ModuleList([
            layer._checkpoint_wrapped_module for layer in encoder.network.encoder.layers])
    set_seed(1)
    targets = []
    if args.mode == "lora_bf16":
        from peft import LoraConfig, inject_adapter_in_model

        targets = [name for name, module in encoder.network.named_modules() if isinstance(module, nn.Linear)]
        if args.unimol_qv_only:
            if args.model != "unimol":
                raise ValueError("QV targets apply only to Uni-Mol")
            targets = [name for name in targets if name.endswith(("self_attn.linear_q", "self_attn.linear_v"))]
        if not targets:
            raise ValueError("No LoRA target layers")
        inject_adapter_in_model(LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                          target_modules=targets, bias="none"), encoder.network)
        assert all(not p.requires_grad for n, p in encoder.network.named_parameters() if "lora_" not in n)
    set_seed(1)
    model = MultiTaskModel(len(data["conditions"]), len(data["buckets"]), 2048, 3, encoder).cuda().train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([
        {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": 1e-5 / 5},
        {"params": model.head.parameters(), "lr": 1e-4 / 5},
        {"params": model.auxiliary.parameters(), "lr": 1e-4 / 5},
    ], weight_decay=1e-4)
    count = args.warmup + args.rounds * args.steps
    batches = loader(graphs, data["splits"]["train"], 32, shuffle=True)
    generator = torch.Generator().manual_seed(1)
    batches.generator = batches.sampler.generator = generator
    main_batches = list(batches)[:count]
    if len(main_batches) != count:
        raise ValueError("Requested more batches than one epoch")
    aux = data["auxiliary"]
    support = torch.bincount(aux["buckets"], minlength=len(data["buckets"]))
    sampler = WeightedRandomSampler(1.0 / support[aux["buckets"]].double(), count * 32,
                                    generator=torch.Generator().manual_seed(10001))
    aux_batches = list(loader(graphs, aux, 32, sampler=sampler, auxiliary=True))
    result = dict(
        schema="adaptation_throughput.v1", model=args.model, mode=args.mode,
        host=socket.gethostname(), gpu=torch.cuda.get_device_name(), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        packages={p: importlib.metadata.version(p) for p in ("torch", "peft", "transformers")},
        source_sha256=sha256_file(Path(__file__)), prepared_sha256=sha256_file(prepared / "manifest.json"),
        features_sha256=sha256_file(cache / "receipt.json"), seed=1, batch_size=32,
        lora=dict(rank=8, alpha=16, dropout=0, target_modules=targets) if targets else None,
        encoder_parameters=sum(p.numel() for p in encoder.parameters()),
        trainable_encoder_parameters=sum(p.numel() for p in encoder.parameters() if p.requires_grad),
        activation_checkpointing=args.model == "unimol" and not args.no_activation_checkpointing, warmup_steps=args.warmup,
        rounds=args.rounds, steps_per_round=args.steps, auxiliary_weight=0.6,
        main_indices=[b[0].tolist() for b in main_batches], auxiliary_indices=[b[0].tolist() for b in aux_batches],
        includes="CPU batch assembly from cached features, transfers, forward, backward, clipping, Adam; synchronized wall time",
        excludes="model loading, feature generation, validation, checkpoint I/O; no test data used",
        measurements=[],
    )
    for setting in ("main", "auxiliary"):
        set_seed(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for step, (indices, conditions, labels) in enumerate(main_batches):
            if step == args.warmup or step > args.warmup and (step - args.warmup) % args.steps == 0:
                torch.cuda.synchronize()
                start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.mode != "full_fp32"):
                embedding = encoder(indices.to(getattr(encoder, "input_device", "cuda")))
                logits = model.main_logits(embedding, conditions.cuda())
                loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.cuda())
            loss.backward()
            aux_loss = torch.zeros((), device="cuda")
            if setting == "auxiliary":
                indices, buckets, targets, binary = aux_batches[step]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.mode != "full_fp32"):
                    embedding = encoder(indices.to(getattr(encoder, "input_device", "cuda")))
                    logits = model.auxiliary(embedding).gather(1, buckets.cuda()[:, None]).squeeze(1)
                    aux_loss = auxiliary_loss(logits.float(), targets.cuda(), binary.cuda())
                (0.6 * aux_loss).backward()
            if not torch.isfinite(loss + aux_loss):
                raise FloatingPointError("Nonfinite loss")
            torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
            optimizer.step()
            losses = [loss.item(), aux_loss.item()]
            if step >= args.warmup and (step + 1 - args.warmup) % args.steps == 0:
                torch.cuda.synchronize()
                durations.append((time.perf_counter() - start) / args.steps)
        gradients = [n for n, p in encoder.named_parameters() if p.grad is not None and bool(p.grad.abs().sum() > 0)]
        assert gradients, "Disconnected encoder gradients"
        if args.mode == "lora_bf16":
            assert all("lora_" in n for n in gradients), "Base encoder was not frozen"
            assert any("lora_B" in n for n in gradients)
        measurement = dict(setting=setting, seconds_per_step=statistics.mean(durations),
                           round_seconds_per_step=durations, sd_seconds=statistics.stdev(durations),
                           main_examples_per_second=32 / statistics.mean(durations),
                           peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                           last_losses=losses, nonzero_encoder_gradient_tensors=len(gradients), finite=True)
        result["measurements"].append(measurement)
        result["gpu_snapshot"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw",
            "--format=csv,noheader"], text=True)
        write_json_atomic(args.output, result)
        print(json.dumps(measurement), flush=True)


if __name__ == "__main__":
    main()
