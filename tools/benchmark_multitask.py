"""Benchmark de CPU/CUDA sobre una copia temporal del run: no modifica sus checkpoints.

python -m tools.benchmark_multitask --config train/config_runpod.yaml --iters 5
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
import time
from pathlib import Path

import torch

from train import multitask
from train.cuda_decisions import DECISION_BACKENDS, CudaDecisionProfile
from train.multitask import ROOT, MultiTrainer
from train.runtime import cpu_budget, load_config


def trainer_class(cfg):
    """Keep the legacy factory injectable, but benchmark each real architecture."""
    if cfg.get("rs4_program"):
        from train.rs4_trainer import RS4V3Trainer
        return RS4V3Trainer
    if cfg.get("model", {}).get("type") == "recurrent_set":
        from train.recurrent_ppo import RecurrentTrainer
        return RecurrentTrainer
    return MultiTrainer


def acceptance(summary, reference=None, minimum_speed_ratio=0.7, max_memory_gib=None):
    """An explicit speed/memory rejection; never adjusts PPO to pass a gate."""
    reasons, ratio = [], None
    if reference is not None:
        baseline = float(reference["steps_per_second"])
        if baseline <= 0:
            raise ValueError("Reference throughput must be positive")
        if summary.get("comparison_contract") != reference.get("comparison_contract"):
            reasons.append("different PPO, environment or frozen-teammate comparison contract")
        ratio = float(summary["steps_per_second"]) / baseline
        if ratio < minimum_speed_ratio:
            reasons.append(f"useful throughput ratio {ratio:.3f} < {minimum_speed_ratio:.3f}")
    memory = summary.get("cuda_peak_allocated_gib", 0.0)
    if max_memory_gib is not None and memory > max_memory_gib:
        reasons.append(f"CUDA allocated memory {memory:.3f} GiB > {max_memory_gib:.3f} GiB")
    return dict(accepted=not reasons, reasons=reasons, useful_speed_ratio=ratio,
                minimum_speed_ratio=minimum_speed_ratio, max_memory_gib=max_memory_gib)


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
        if hasattr(trainer, "program"):
            # The v3 trainer imports these helpers into its own module; wrapping
            # multitask alone misses the actual calls made by this path.
            from train import rs4_trainer
            self.wrap(rs4_trainer, "scripted_actions", "bots CPU (dentro de decisiones)")
            self.wrap(rs4_trainer, "batch_to_device", "lote PPO: empaquetado/transferencia (dentro de preparación)")
            self.wrap(trainer.inference, "infer", "inferencia agrupada host (dentro de decisiones/bootstrap)")
            for slot in trainer.slots:
                self.wrap(slot.env.cohorts, "begin", "cohortes: apertura (dentro de entorno)")
                self.wrap(slot.env.cohorts, "update", "cohortes: resolución (dentro de entorno)")
        self.wrap(trainer.model, "update_norm", "normalización (dentro de preparación)")
        for method, label in (("assign_modes", "reparto de rivales (setup)"),
                              ("_prepare_policy_groups", "índices de rivales (setup)"),
                              ("opponent_curriculum", "currículo de rivales (mantenimiento)"),
                              ("log", "logging completo"),
                              ("maybe_rebalance_or_advance", "reparto/avance de etapa")):
            self.wrap(trainer, method, label)
        for slot in trainer.slots:
            self.wrap(slot.env, "step", "entorno total")
            self.wrap(slot.env, "step", f"entorno {slot.task.name} (dentro de entorno total)")
            self.wrap(slot.env, "observe", "observaciones (dentro de entorno)")
            self.wrap(slot.env, "_potentials", "potenciales (dentro de entorno)")
            for method, label in (("_team_ball_dist", "distancia por equipo"),
                                  ("_spread_potential", "separación por equipo"),
                                  ("_advance_pending_passes", "retención de pases"),
                                  ("_cooperation_touch_reward", "contactos/pases"),
                                  ("_setpiece_pre_tick", "protección de saques"),
                                  ("_setpiece_post_tick", "finalización de saques"),
                                  ("_rs4_potential", "formación/amenaza RS4"),
                                  ("_rs4_restart_potential", "aproximación a saques RS4"),
                                  ("_record_defensive_out_touch", "contactos defensivos RS4")):
                self.wrap(slot.env, method, f"{label} (dentro de entorno)")
            self.wrap(slot.env.sim, "step", "física (dentro de entorno)")
            self.wrap(slot.env.sim, "step_frames", "física (dentro de entorno)")
            self.wrap(slot.env.sim, "step_frames_with_touches", "física (dentro de entorno)")

    def wrap(self, owner, attr, label):
        original = getattr(owner, attr, None)
        if original is None:  # Different trainers expose different optimized paths.
            return
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
    ap.add_argument("--no-reward-geometry", action="store_true", help="Geometría de rewards NumPy para comparar, sin cambiar física/red/PPO")
    ap.add_argument("--no-callbacks", action="store_true", help="Callbacks de saques/cooperación de referencia; conserva geometría y CUDA")
    ap.add_argument("--no-optimize-rs4", action="store_true", help="Callbacks RS4 NumPy y formación en lote completo; conserva rewards/PPO")
    ap.add_argument("--no-reuse-capture-pool", action="store_true", help="Crear stream/pool nuevos en cada captura CUDA para comparar")
    ap.add_argument("--no-reuse-minibatch-obs", action="store_true", help="Asignar observaciones nuevas en cada minibatch CUDA para comparar")
    ap.add_argument("--no-reuse-ppo-batch", action="store_true", help="Preparación anterior: asignar/pinear cada lote para comparar")
    ap.add_argument("--no-cache-bc-logits", action="store_true", help="Recalcular referencia BC en cada época PPO para comparar CPU")
    ap.add_argument("--no-optimize-cpu", action="store_true", help="Desactivar agrupamiento CPU y buffer de minibatch; conserva cache BC")
    ap.add_argument("--json-output", type=Path, help="Guardar resumen para barridos de hilos")
    ap.add_argument("--reference-json", type=Path, help="Rechazar diferencias de mezcla/PPO o pérdida de velocidad excesiva")
    ap.add_argument("--minimum-speed-ratio", type=float, default=0.7)
    ap.add_argument("--max-memory-gib", type=float, help="Límite explícito de VRAM asignada, sin reducir PPO")
    ap.add_argument("--companion-fraction", type=float, help="Misma fracción de equipos mixtos para comparar candidatos RS4 v3")
    args = ap.parse_args()
    if args.iters < 1 or args.warmup < 1:
        ap.error("iters y warmup deben ser positivos (Numba necesita calentamiento)")
    if args.numba_threads is not None and args.numba_threads < 1:
        ap.error("numba-threads debe ser positivo")
    if args.torch_threads is not None and args.torch_threads < 1:
        ap.error("torch-threads debe ser positivo")
    if args.baseline and args.decision_backend not in (None, "legacy"):
        ap.error("--baseline usa decisiones legacy; comparar sólo inferencia sin --baseline")
    if not 0 < args.minimum_speed_ratio <= 1:
        ap.error("minimum-speed-ratio debe estar en (0, 1]")
    if args.max_memory_gib is not None and args.max_memory_gib <= 0:
        ap.error("max-memory-gib debe ser positivo")
    if args.companion_fraction is not None and not 0 <= args.companion_fraction <= 1:
        ap.error("companion-fraction debe estar entre 0 y 1")
    cfg = copy.deepcopy(load_config(args.config))
    if args.device:
        cfg["ppo"]["device"] = args.device
    if args.torch_threads is not None:
        cfg["ppo"]["torch_threads"] = args.torch_threads
    if args.numba_threads is not None:
        cfg["ppo"]["numba_threads"] = args.numba_threads
    cfg.setdefault("runtime", {}).setdefault("optimize_rollout", True)
    cfg["runtime"]["benchmark"] = True
    if cfg.get("rs4_program"):
        # A pilot-end checkpoint has already reached its original 200M segment.
        # Extend only this disposable copy's segment, never source config/ledger.
        program = cfg["rs4_program"]
        cfg["ppo"]["total_steps"] = program["start_steps"] + program["branch_budget_steps"]
    if args.companion_fraction is not None:
        if not cfg.get("rs4_program"):
            ap.error("companion-fraction requiere un programa RS4 v3")
        cfg["runtime"]["benchmark_companion_fraction"] = args.companion_fraction
    if args.baseline:
        cfg["runtime"]["optimize_rollout"] = False
        cfg["runtime"]["optimize_cpu"] = False
    if args.no_cache_bc_logits:
        cfg["runtime"]["cache_bc_logits_cpu"] = False
        cfg["runtime"]["cache_bc_logits"] = False
    if args.no_optimize_cpu:
        cfg["runtime"]["optimize_cpu"] = False
    if args.no_reward_geometry:
        cfg["runtime"]["optimize_reward_geometry"] = False
    if args.no_callbacks:
        cfg["runtime"]["optimize_callbacks"] = False
    if args.no_optimize_rs4:
        cfg["runtime"]["optimize_rs4"] = False
    if args.no_reuse_capture_pool:
        cfg["runtime"]["reuse_cuda_capture_pool"] = False
    if args.no_reuse_minibatch_obs:
        cfg["runtime"]["reuse_minibatch_obs"] = False
    if args.profile_rollout:
        cfg["runtime"]["profile_update"] = True
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
    cfg["log"].pop("checkpoint_interval_seconds", None)
    cfg["log"].pop("checkpoint_history_interval_seconds", None)
    cfg["league"]["snapshot_every"] = 10**12
    cfg["schedule"]["rebalance_every"] = 10**12
    (ROOT / "runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_benchmark_", dir=ROOT / "runs") as directory:
        shutil.copy2(args.checkpoint, Path(directory) / "latest.pt")
        saved_config = Path(args.checkpoint).parent / "config.yaml"
        if cfg["runtime"].get("preserve_bc_reference", False) and saved_config.exists():
            shutil.copy2(saved_config, Path(directory) / "config.yaml")
        trainer = trainer_class(cfg)(cfg, Path(directory).name, resume=True)
        profile = RolloutProfile(trainer) if args.profile_rollout else None
        decision_profile = (CudaDecisionProfile(trainer.device)
                            if args.profile_rollout and trainer.device.type == "cuda" else None)
        trainer._decision_profile = decision_profile
        records = []
        batch_stats = []
        iteration_timings, update_timings, learning_stats = [], [], []
        original_log = trainer.log

        def log(stats, lr, n, rollout, update):
            records.append((n, rollout, stats.get("timing/prepare_seconds", 0.0), update,
                            stats.get("timing/cuda_capture_seconds", stats.get("cuda_graph/capture_seconds", 0.0)),
                            stats.get("runtime/cuda_graph_captures", stats.get("cuda_graph/captures", 0)),
                            stats.get("runtime/cuda_graph_replays", stats.get("cuda_graph/replays", 0))))
            batch_stats.append({key: value for key, value in stats.items()
                                if key.startswith("timing/ppo_batch_") or key == "runtime/ppo_batch_allocations"})
            update_timings.append({key: value for key, value in stats.items()
                                   if key.startswith("timing/update_") and key != "timing/update_seconds"})
            learning_stats.append({key: stats.get(key, 0.0) for key in ("entropy", "approx_kl", "clipfrac", "bc_kl")})
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
                    cpu_start = time.process_time()
                trainer.iterate()
                iteration_timings.append(trainer._last_iteration_timings)
            elapsed = time.perf_counter() - start
            cpu_seconds = time.process_time() - cpu_start
            cpu_cores = cpu_seconds / elapsed
            budget = cpu_budget()
            measured = records[args.warmup:]
            samples = sum(row[0] for row in measured)
            print(f"\n{samples / elapsed:,.0f} pasos/s reales | {elapsed / args.iters:.3f} s/iter")
            print(f"CPU del proceso: {cpu_cores:.2f} núcleos equivalentes de {budget} disponibles "
                  f"({100 * cpu_cores / budget:.1f}% del cupo; no el porcentaje del panel del host)")
            print(f"muestras útiles: {samples:,} total | {samples / args.iters:,.1f}/iter "
                  f"| mínimo {min(row[0] for row in measured):,} | máximo {max(row[0] for row in measured):,}")
            if hasattr(trainer, "program"):
                simulated = args.iters * cfg["ppo"]["rollout_len"] * trainer.total_rows
                print(f"filas simuladas: {simulated:,} | fracción aprendiz {samples / simulated:.1%}")
            for index, name in ((1, "rollout"), (2, "preparación/GAE"), (3, "update")):
                print(f"{name}: {sum(row[index] for row in measured) / args.iters:.3f} s/iter")
            extras = {key: sum(row[key] for row in iteration_timings[args.warmup:]) / args.iters
                      for key in ("setup", "maintenance", "logging", "schedule")}
            extras["return_overhead"] = (elapsed / args.iters - sum(extras.values())
                                         - sum(sum(row[1:4]) for row in measured) / args.iters)
            print("fuera de las fases: " + " | ".join(f"{k} {v:.3f}s" for k, v in extras.items()))
            metrics = {key: sum(row[key] for row in learning_stats[args.warmup:]) / args.iters
                       for key in learning_stats[-1]}
            print("PPO (media): " + " | ".join(f"{k} {v:.5f}" for k, v in metrics.items()))
            from numba import get_num_threads
            summary = dict(samples=samples, elapsed_seconds=elapsed, steps_per_second=samples / elapsed,
                           warmup_samples=sum(row[0] for row in records[:args.warmup]),
                           diagnostic_ppo_samples=sum(row[0] for row in records),
                           samples_per_iteration=[row[0] for row in measured],
                           rollout_seconds=sum(row[1] for row in measured) / args.iters,
                           prepare_seconds=sum(row[2] for row in measured) / args.iters,
                           update_seconds=sum(row[3] for row in measured) / args.iters,
                           torch_threads=torch.get_num_threads(), numba_threads=get_num_threads(),
                           device=str(trainer.device), config=str(args.config), checkpoint=str(args.checkpoint),
                           warmup=args.warmup, iters=args.iters, runtime=cfg.get("runtime", {}),
                           extra_seconds=extras, learning_metrics=metrics)
            summary["model_config"] = trainer.model.config() if hasattr(trainer, "model") else cfg.get("model", {})
            summary["comparison_contract"] = {
                "env": cfg.get("env", {}),
                "ppo": {key: cfg.get("ppo", {}).get(key) for key in
                        ("rollout_len", "epochs", "minibatch", "gamma", "gae_lambda", "clip")},
                "frozen_teammates": cfg["runtime"].get("benchmark_companion_fraction",
                                                        cfg.get("rs4_program", {}).get("phases", [])),
                "seed": cfg.get("seed"),
                "precision": dict(dtype="float32", amp=False,
                                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
            }
            summary["cuda_peak_allocated_gib"] = (torch.cuda.max_memory_allocated(trainer.device) / 2**30
                                                  if trainer.device.type == "cuda" else 0.0)
            summary["cuda_peak_reserved_gib"] = (torch.cuda.max_memory_reserved(trainer.device) / 2**30
                                                 if trainer.device.type == "cuda" else 0.0)
            if hasattr(trainer, "program"):
                summary["simulated_samples"] = simulated
                summary["learner_fraction"] = samples / simulated
                settings = trainer.program.settings()
                summary["rs4_profile"] = {key: settings[key] for key in
                    ("phase_id", "exercise_fraction", "frozen_teammates_fraction", "opponent_mix")}
                if args.companion_fraction is not None:
                    summary["rs4_profile"]["frozen_teammates_fraction"] = args.companion_fraction
            summary.update(cpu_seconds=cpu_seconds, cpu_core_equivalents=cpu_cores,
                           cpu_budget=budget, cpu_budget_percent=100 * cpu_cores / budget)
            summary.update(cuda_capture_seconds=sum(row[4] for row in measured) / args.iters,
                           cuda_captures=sum(row[5] for row in measured) / args.iters,
                           cuda_replays=sum(row[6] for row in measured) / args.iters)
            if profile:
                summary["update_profile_seconds"] = {
                    key: sum(row[key] for row in update_timings[args.warmup:]) / args.iters
                    for key in update_timings[-1]}
            reference = json.loads(args.reference_json.read_text(encoding="utf-8")) if args.reference_json else None
            summary["acceptance"] = acceptance(summary, reference, args.minimum_speed_ratio, args.max_memory_gib)
            summary["memory_gate_passed"] = (args.max_memory_gib is not None
                                              and summary["cuda_peak_allocated_gib"] <= args.max_memory_gib)
            if args.json_output:
                args.json_output.parent.mkdir(parents=True, exist_ok=True)
                args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if trainer.device.type == "cuda":
                print(f"VRAM máxima asignada: {torch.cuda.max_memory_allocated(trainer.device) / 2**30:.2f} GiB")
                captures, replays = (sum(row[i] for row in measured) / args.iters for i in (5, 6))
                print(f"decisiones CUDA: {trainer.cuda_decisions} | capturas {captures:.1f}/iter | replays {replays:.0f}/iter")
                print(f"preparación/captura graph: {sum(row[4] for row in measured) / args.iters:.3f} s/iter (incluida en rollout)")
                decision_engine = getattr(trainer, "inference", getattr(trainer, "_decision_graph", None))
                if decision_engine is not None and decision_engine.disabled_reason is not None:
                    print(f"fallback eager: {decision_engine.disabled_reason}")
                print(f"buffers PPO reutilizados: {trainer._batch_transfer is not None}")
                print(f"pool/stream de captura reutilizados: {cfg['runtime'].get('reuse_cuda_capture_pool', True)} "
                      f"| obs de minibatch reutilizadas: {cfg['runtime'].get('reuse_minibatch_obs', True)}")
                if trainer._batch_transfer is not None:
                    for key in batch_stats[-1]:
                        average = sum(row[key] for row in batch_stats[args.warmup:]) / args.iters
                        print(f"  {key}: {average:.4f}/iter")
            if profile:
                profile.report(args.iters)
                print("\nDesglose update: host y GPU stream NO se suman (incluido en update).")
                print("GPU stream incluye huecos de envío CPU; no mide saturación ni sólo kernels.")
                for key in update_timings[-1]:
                    average = sum(row[key] for row in update_timings[args.warmup:]) / args.iters
                    print(f"  {key}: {average:.3f} s/iter")
            if decision_profile:
                if trainer.cuda_decisions == "legacy":
                    print("\nNota: en legacy, inferencia también incluye los muestreos; su fase muestreo no es comparable por separado.")
                decision_profile.report(args.iters)
            if not summary["acceptance"]["accepted"]:
                raise SystemExit("Benchmark rechazado: " + "; ".join(summary["acceptance"]["reasons"]))
        finally:
            trainer._decision_profile = None
            if profile:
                profile.close()
            trainer.writer.close()


if __name__ == "__main__":
    main()
