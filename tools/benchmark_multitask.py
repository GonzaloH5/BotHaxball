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

from train import multitask
from train.cuda_decisions import DECISION_BACKENDS, CudaDecisionProfile
from train.multitask import ROOT, MultiTrainer
from train.runtime import load_config


class RolloutProfile:
    """Tiempos inclusivos opcionales; sin sincronizaciones CUDA adicionales.

    act incluye bots/espera GPU, step incluye física/obs/potenciales. No sumar padres e hijos.
    """
    def __init__(self, trainer):
        self.totals, self.calls, self.originals = {}, {}, []
        self.wrap(trainer, "act", "decisiones (red+transferencia+bots)")
        if trainer.device.type == "cuda" and trainer.cuda_decisions != "legacy":
            self.wrap(trainer, "_cuda_inference", "host inferencia/captura (dentro de decisiones)")
            self.wrap(trainer, "_sample_cuda", "host muestreo (dentro de decisiones)")
        self.wrap(multitask, "scripted_actions", "bots CPU (dentro de decisiones)")
        self.wrap(trainer, "values_many", "bootstrap de valores")
        self.wrap(multitask, "batch_to_device", "lote PPO: empaquetado/transferencia (dentro de preparación)")
        self.wrap(trainer.model, "update_norm", "normalización (dentro de preparación)")
        for slot in trainer.slots:
            self.wrap(slot.env, "step", "entorno total")
            self.wrap(slot.env, "observe", "observaciones (dentro de entorno)")
            self.wrap(slot.env, "_potentials", "potenciales (dentro de entorno)")
            self.wrap(slot.env.sim, "step", "física (dentro de entorno)")
            self.wrap(slot.env.sim, "step_frames", "física (dentro de entorno)")

    def wrap(self, owner, attr, label):
        original = getattr(owner, attr)
        self.originals.append((owner, attr, original))

        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.totals[label] = self.totals.get(label, 0.0) + time.perf_counter() - start
                self.calls[label] = self.calls.get(label, 0) + 1
        setattr(owner, attr, measured)

    def reset(self):
        self.totals.clear()
        self.calls.clear()

    def report(self, iters):
        print("\nDesglose inclusivo (padres/hijos y CPU/GPU solapados NO se suman):")
        for label, total in self.totals.items():
            print(f"  {label}: {total / iters:.3f} s/iter | {self.calls[label] / iters:.0f} llamadas/iter")

    def close(self):
        for owner, attr, original in reversed(self.originals):
            setattr(owner, attr, original)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="train/config_runpod.yaml")
    ap.add_argument("--checkpoint", default="runs/multi/latest.pt")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", choices=("cpu", "cuda"))
    ap.add_argument("--torch-threads", type=int, help="Hilos CPU de Torch (inferencia y update)")
    ap.add_argument("--numba-threads", type=int)
    ap.add_argument("--baseline", action="store_true", help="Ruta de referencia sin optimizaciones del rollout")
    ap.add_argument("--decision-backend", choices=DECISION_BACKENDS,
                    help="legacy: decisiones anteriores; eager: un muestreo; auto/graph: CUDA Graph (graph exige captura)")
    ap.add_argument("--profile-rollout", action="store_true", help="Desglose inclusivo por componente (añade overhead)")
    ap.add_argument("--no-reuse-ppo-batch", action="store_true", help="Preparación anterior: asignar/pinear cada lote para comparar")
    args = ap.parse_args()
    if args.iters < 1 or args.warmup < 1:
        ap.error("iters y warmup deben ser positivos (Numba necesita calentamiento)")
    if args.numba_threads is not None and args.numba_threads < 1:
        ap.error("numba-threads debe ser positivo")
    if args.torch_threads is not None and args.torch_threads < 1:
        ap.error("torch-threads debe ser positivo")
    if args.baseline and args.decision_backend not in (None, "legacy"):
        ap.error("--baseline usa decisiones legacy; comparar sólo inferencia sin --baseline")
    cfg = copy.deepcopy(load_config(args.config))
    if args.device:
        cfg["ppo"]["device"] = args.device
    if args.torch_threads is not None:
        cfg["ppo"]["torch_threads"] = args.torch_threads
    if args.numba_threads is not None:
        cfg["ppo"]["numba_threads"] = args.numba_threads
    cfg.setdefault("runtime", {}).setdefault("optimize_rollout", True)
    if args.baseline:
        cfg["runtime"]["optimize_rollout"] = False
    if args.baseline or args.no_reuse_ppo_batch:
        cfg["runtime"]["reuse_ppo_batch"] = False
    else:
        cfg["runtime"].setdefault("reuse_ppo_batch", True)
    if args.baseline:
        cfg["runtime"]["cuda_decisions"] = "legacy"
    elif args.decision_backend:
        cfg["runtime"]["cuda_decisions"] = args.decision_backend
    # Sin replays/procesos externos ni checkpoints periódicos durante la medición.
    cfg["log"].update(every=1, checkpoint_every=10**12, replay_every=0)
    cfg["league"]["snapshot_every"] = 10**12
    cfg["schedule"]["rebalance_every"] = 10**12
    (ROOT / "runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_benchmark_", dir=ROOT / "runs") as directory:
        shutil.copy2(args.checkpoint, Path(directory) / "latest.pt")
        trainer = MultiTrainer(cfg, Path(directory).name, resume=True)
        profile = RolloutProfile(trainer) if args.profile_rollout else None
        decision_profile = (CudaDecisionProfile(trainer.device)
                            if args.profile_rollout and trainer.device.type == "cuda" else None)
        trainer._decision_profile = decision_profile
        records = []
        batch_stats = []
        original_log = trainer.log

        def log(stats, lr, n, rollout, update):
            records.append((n, rollout, stats.get("timing/prepare_seconds", 0.0), update,
                            stats.get("timing/cuda_capture_seconds", 0.0),
                            stats.get("runtime/cuda_graph_captures", 0),
                            stats.get("runtime/cuda_graph_replays", 0)))
            batch_stats.append({key: value for key, value in stats.items()
                                if key.startswith("timing/ppo_batch_") or key == "runtime/ppo_batch_allocations"})
            original_log(stats, lr, n, rollout, update)

        trainer.log = log
        start = None
        try:
            for iteration in range(args.warmup + args.iters):
                if iteration == args.warmup:
                    if trainer.device.type == "cuda":
                        torch.cuda.synchronize(trainer.device)
                        torch.cuda.reset_peak_memory_stats(trainer.device)
                    if profile:
                        profile.reset()
                    if decision_profile:
                        decision_profile.reset()
                    start = time.perf_counter()
                trainer.iterate()
            elapsed = time.perf_counter() - start
            measured = records[args.warmup:]
            samples = sum(row[0] for row in measured)
            print(f"\n{samples / elapsed:,.0f} pasos/s reales | {elapsed / args.iters:.3f} s/iter")
            print(f"muestras útiles: {samples:,} total | {samples / args.iters:,.1f}/iter "
                  f"| mínimo {min(row[0] for row in measured):,} | máximo {max(row[0] for row in measured):,}")
            for index, name in ((1, "rollout"), (2, "preparación/GAE"), (3, "update")):
                print(f"{name}: {sum(row[index] for row in measured) / args.iters:.3f} s/iter")
            if trainer.device.type == "cuda":
                print(f"VRAM máxima asignada: {torch.cuda.max_memory_allocated(trainer.device) / 2**30:.2f} GiB")
                captures, replays = (sum(row[i] for row in measured) / args.iters for i in (5, 6))
                print(f"decisiones CUDA: {trainer.cuda_decisions} | capturas {captures:.1f}/iter | replays {replays:.0f}/iter")
                print(f"preparación/captura graph: {sum(row[4] for row in measured) / args.iters:.3f} s/iter (incluida en rollout)")
                if trainer._decision_graph is not None and trainer._decision_graph.disabled_reason is not None:
                    print(f"fallback eager: {trainer._decision_graph.disabled_reason}")
                print(f"buffers PPO reutilizados: {trainer._batch_transfer is not None}")
                if trainer._batch_transfer is not None:
                    for key in batch_stats[-1]:
                        average = sum(row[key] for row in batch_stats[args.warmup:]) / args.iters
                        print(f"  {key}: {average:.4f}/iter")
            if profile:
                profile.report(args.iters)
            if decision_profile:
                if trainer.cuda_decisions == "legacy":
                    print("\nNota: en legacy, inferencia también incluye los muestreos; su fase muestreo no es comparable por separado.")
                decision_profile.report(args.iters)
        finally:
            trainer._decision_profile = None
            if profile:
                profile.close()
            trainer.writer.close()


if __name__ == "__main__":
    main()
