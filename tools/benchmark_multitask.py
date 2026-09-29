"""Benchmark de CPU/CUDA sobre una copia temporal del run: no modifica sus checkpoints.

python -m tools.benchmark_multitask --config train/config_runpod.yaml --iters 5
"""
from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
import time
from pathlib import Path

import torch

from train.multitask import ROOT, MultiTrainer
from train.runtime import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="train/config_runpod.yaml")
    ap.add_argument("--checkpoint", default="runs/multi/latest.pt")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", choices=("cpu", "cuda"))
    ap.add_argument("--numba-threads", type=int)
    args = ap.parse_args()
    if args.iters < 1 or args.warmup < 1:
        ap.error("iters y warmup deben ser positivos (Numba necesita calentamiento)")
    if args.numba_threads is not None and args.numba_threads < 1:
        ap.error("numba-threads debe ser positivo")
    cfg = copy.deepcopy(load_config(args.config))
    if args.device:
        cfg["ppo"]["device"] = args.device
    if args.numba_threads is not None:
        cfg["ppo"]["numba_threads"] = args.numba_threads
    # Sin replays/procesos externos ni checkpoints periódicos durante la medición.
    cfg["log"].update(every=1, checkpoint_every=10**12, replay_every=0)
    cfg["league"]["snapshot_every"] = 10**12
    cfg["schedule"]["rebalance_every"] = 10**12
    (ROOT / "runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_benchmark_", dir=ROOT / "runs") as directory:
        shutil.copy2(args.checkpoint, Path(directory) / "latest.pt")
        trainer = MultiTrainer(cfg, Path(directory).name, resume=True)
        records = []
        original_log = trainer.log

        def log(stats, lr, n, rollout, update):
            records.append((n, rollout, stats.get("timing/prepare_seconds", 0.0), update))
            original_log(stats, lr, n, rollout, update)

        trainer.log = log
        start = None
        try:
            for iteration in range(args.warmup + args.iters):
                if iteration == args.warmup:
                    if trainer.device.type == "cuda":
                        torch.cuda.synchronize(trainer.device)
                        torch.cuda.reset_peak_memory_stats(trainer.device)
                    start = time.perf_counter()
                trainer.iterate()
            elapsed = time.perf_counter() - start
            measured = records[args.warmup:]
            samples = sum(row[0] for row in measured)
            print(f"\n{samples / elapsed:,.0f} pasos/s reales | {elapsed / args.iters:.3f} s/iter")
            for index, name in ((1, "rollout"), (2, "preparación/GAE"), (3, "update")):
                print(f"{name}: {sum(row[index] for row in measured) / args.iters:.3f} s/iter")
            if trainer.device.type == "cuda":
                print(f"VRAM máxima asignada: {torch.cuda.max_memory_allocated(trainer.device) / 2**30:.2f} GiB")
        finally:
            trainer.writer.close()


if __name__ == "__main__":
    main()
