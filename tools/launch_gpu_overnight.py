"""Elegir hilos de física automáticamente y reanudar el perfil nocturno CUDA.

python -m tools.launch_gpu_overnight --run multi
Los sondeos entrenan copias temporales; luego se reanuda el checkpoint original.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from train.runtime import cpu_budget, load_config

ROOT = Path(__file__).resolve().parents[1]


def choose_result(results, baseline):
    best = max(results, key=lambda r: r["steps_per_second"])
    original = next((r for r in results if r["numba_threads"] == baseline), None)
    # Un sondeo corto no justifica cambiar hilos por una diferencia mínima.
    if original and best["steps_per_second"] < original["steps_per_second"] * 1.03:
        return original
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="multi")
    ap.add_argument("--config", default="train/config_gpu_overnight.yaml")
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 6, 8, 10])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--skip-tuning", action="store_true", help="Reanudar directamente con los hilos del perfil")
    args = ap.parse_args()
    if min(*args.threads, args.warmup, args.iters) < 1:
        ap.error("hilos, warmup e iters deben ser positivos")
    import torch
    if not torch.cuda.is_available():
        ap.error("Este lanzador requiere CUDA disponible; activar el entorno PyTorch del Pod")
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    if cfg["ppo"]["device"] != "cuda":
        ap.error("El perfil debe usar ppo.device=cuda")
    checkpoint = ROOT / "runs" / args.run / "latest.pt"
    if not checkpoint.is_file():
        ap.error(f"No existe {checkpoint}; no se inicia un modelo desde cero")
    from numba import config as numba_config
    limit = min(cpu_budget(), numba_config.NUMBA_NUM_THREADS)
    baseline = min(int(cfg["ppo"]["numba_threads"]), limit)
    selected = baseline
    results = []
    if not args.skip_tuning:
        report_root = ROOT / "reports" / "gpu_startup"
        report_root.mkdir(parents=True, exist_ok=True)
        report = Path(tempfile.mkdtemp(prefix="probe_", dir=report_root))
        candidates = list(dict.fromkeys([baseline] + [n for n in args.threads if n <= limit]))
        print(f"Sondeo CUDA: física {candidates}, Torch auto, cupo CPU {limit}. Resultados: {report}", flush=True)
        with tempfile.TemporaryDirectory(prefix="_gpu_startup_", dir=ROOT / "runs") as temporary:
            frozen = Path(temporary) / "latest.pt"
            shutil.copy2(checkpoint, frozen)
            saved_cfg = checkpoint.parent / "config.yaml"
            if saved_cfg.exists():
                shutil.copy2(saved_cfg, Path(temporary) / "config.yaml")
            def measure(n, phase):
                output = report / f"{phase}_physics_{n}.json"
                command = [sys.executable, "-u", "-m", "tools.benchmark_multitask",
                           "--config", str(cfg_path), "--checkpoint", str(frozen),
                           "--numba-threads", str(n), "--warmup", str(args.warmup),
                           "--iters", str(args.iters), "--json-output", str(output)]
                # Un fallo de entrenamiento/CUDA se conserva y detiene el lanzador;
                # no ocultar errores empezando una sesión potencialmente averiada.
                subprocess.run(command, cwd=ROOT, check=True)
                result = json.loads(output.read_text(encoding="utf-8"))
                results.append(result)
                return result

            initial = [measure(n, "initial") for n in candidates]
            best = choose_result(initial, baseline)
            if best["numba_threads"] != baseline:
                # Confirmar en orden inverso antes de adoptar otro reparto.
                finalist = measure(best["numba_threads"], "confirm")
                original = measure(baseline, "confirm")
                best = choose_result([finalist, original], baseline)
        (report / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        selected = best["numba_threads"]
        (report / "selected.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(f"Seleccionado: física {selected}, {best['steps_per_second']:,.0f} pasos/s reales en el sondeo.", flush=True)
    print("Reanudando entrenamiento nocturno desde el checkpoint original...", flush=True)
    command = [sys.executable, "-u", "-m", "train.multitask", "--config", str(cfg_path),
               "--run", args.run, "--resume", "--override", f"ppo.numba_threads={selected}"]
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        # En un terminal foreground el hijo recibe el mismo SIGINT/Ctrl+C:
        # esperar su guardado, no enviarlo por segunda vez.
        code = process.wait()
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
