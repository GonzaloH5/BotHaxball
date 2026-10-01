"""Barrido secuencial CPU: física, Torch y confirmación, sobre un checkpoint fijo.

python -m tools.tune_cpu --config train/config_cpu.yaml --warmup 3 --iters 15
No modifica configs ni checkpoints; cada prueba corre en un proceso nuevo.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def tune(measure, candidates, torch_threads, numba_threads):
    """Descenso por coordenadas: un parámetro por vez, más repetición inversa."""
    baseline = measure(torch_threads, numba_threads, "baseline")
    physics = [baseline]
    for n in candidates:
        if n != numba_threads:
            physics.append(measure(torch_threads, n, "physics"))
    best_physics = max(physics, key=lambda r: r["steps_per_second"])
    n = best_physics["numba_threads"]
    torch_runs = [best_physics]
    for t in candidates:
        if t != torch_threads:
            torch_runs.append(measure(t, n, "torch"))
    best = max(torch_runs, key=lambda r: r["steps_per_second"])
    # El finalista se repite antes de la baseline para detectar ruido/deriva.
    confirmation = measure(best["torch_threads"], best["numba_threads"], "confirm_best")
    baseline_repeat = measure(torch_threads, numba_threads, "confirm_baseline")
    return baseline, best, confirmation, baseline_repeat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="train/config_cpu.yaml")
    ap.add_argument("--checkpoint", default="runs/multi/latest.pt")
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--numba-threads", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--out", default="reports/cpu_tuning")
    args = ap.parse_args()
    if min(*args.threads, args.torch_threads, args.numba_threads, args.iters, args.warmup) < 1:
        ap.error("hilos, warmup e iters deben ser positivos")
    root = Path(__file__).resolve().parents[1]
    config = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    # Directorio nuevo: nunca sobrescribir mediciones anteriores.
    output_root = Path(args.out).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="sweep_", dir=output_root))
    print(f"Resultados: {output}", flush=True)
    results = []
    (root / "runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_cpu_tuning_", dir=root / "runs") as temp:
        frozen = Path(temp) / "latest.pt"
        shutil.copy2(checkpoint, frozen)

        def measure(t, n, phase):
            path = output / f"{len(results):02d}_{phase}_torch{t}_numba{n}.json"
            command = [sys.executable, "-u", "-m", "tools.benchmark_multitask", "--device", "cpu",
                       "--config", str(config), "--checkpoint", str(frozen),
                       "--torch-threads", str(t), "--numba-threads", str(n),
                       "--warmup", str(args.warmup), "--iters", str(args.iters),
                       "--json-output", str(path)]
            print(f"\n{phase}: Torch {t}, física {n}", flush=True)
            subprocess.run(command, cwd=root, check=True)
            result = json.loads(path.read_text(encoding="utf-8"))
            result["phase"] = phase
            results.append(result)
            (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            return result

        baseline, best, confirmation, repeated = tune(
            measure, list(dict.fromkeys(args.threads)), args.torch_threads, args.numba_threads)
    print("\nComparación (pasos útiles/s reales):")
    for r in results:
        print(f"{r['phase']:18s} Torch {r['torch_threads']:2d} física {r['numba_threads']:2d}: "
              f"{r['steps_per_second']:,.0f}/s | roll {r['rollout_seconds']:.3f}s "
              f"prep {r['prepare_seconds']:.3f}s upd {r['update_seconds']:.3f}s")
    first_gain = best["steps_per_second"] / baseline["steps_per_second"] - 1
    repeated_gain = confirmation["steps_per_second"] / repeated["steps_per_second"] - 1
    print(f"Mejora inicial {first_gain:+.1%}; repetida {repeated_gain:+.1%}.")
    if len({tuple(r["samples_per_iteration"]) for r in results}) > 1:
        print("Las muestras por iteración variaron: revisar trayectorias/currículo antes de adoptar el resultado.")
    print(f"Finalista: ppo.torch_threads={best['torch_threads']} ppo.numba_threads={best['numba_threads']}.")
    print("El barrido no cambia tu configuración. Revisar estabilidad de la repetición antes de adoptarlo.")


if __name__ == "__main__":
    main()
